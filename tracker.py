# =========================================================================
# BOULDER OSMP TRAIL TRACKER
# Pulls your Strava history, compares it to the OSMP trail network, and
# publishes the map to docs/index.html (served by GitHub Pages).
#
# Runs automatically via .github/workflows/update-map.yml.
# Strava credentials come from the repo's Actions secrets — never put them in this file.
# =========================================================================

import os
import sys
import time
import hashlib
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import polyline
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import LocateControl
from shapely.geometry import LineString
from shapely.ops import unary_union
from branca.element import MacroElement
from jinja2 import Template

ROOT = Path(__file__).resolve().parent
GEOJSON_PATH = ROOT / "osmp_trails.geojson"   # replace this file in the repo to update trail data
OUT_DIR = ROOT / "docs"                        # GitHub Pages publishes this folder
MAP_PATH = OUT_DIR / "index.html"
CSV_PATH = OUT_DIR / "boulder_challenge_progress_report.csv"
FINGERPRINT_PATH = OUT_DIR / "fingerprint.txt"
LOCAL_TZ = ZoneInfo("America/Denver")
ACTIVITY_TYPES = ['Run', 'Hike', 'Trail Run']


# =========================================================================
# PART 1: STRAVA (replaces Cell 1 — silent auth via refresh token)
# =========================================================================

def get_access_token():
    missing = [k for k in ("STRAVA_CLIENT_ID", "STRAVA_CLIENT_SECRET", "STRAVA_REFRESH_TOKEN") if not os.environ.get(k)]
    if missing:
        sys.exit(f"Missing secrets: {', '.join(missing)}. Add them under Settings → Secrets and variables → Actions.")

    # .strip() removes stray spaces, quotes or line breaks picked up when pasting secrets
    creds = {k: os.environ[k].strip().strip('"').strip("'").strip()
             for k in ("STRAVA_CLIENT_ID", "STRAVA_CLIENT_SECRET", "STRAVA_REFRESH_TOKEN")}

    res = requests.post("https://www.strava.com/oauth/token", data={
        "client_id": creds["STRAVA_CLIENT_ID"],
        "client_secret": creds["STRAVA_CLIENT_SECRET"],
        "grant_type": "refresh_token",
        "refresh_token": creds["STRAVA_REFRESH_TOKEN"],
    }, timeout=30)
    try:
        data = res.json()
    except ValueError:
        data = {"message": res.text[:200]}
    if "access_token" not in data:
        # Strava names the rejected field (e.g. refresh_token / client_secret) without echoing its value
        problems = "; ".join(f"{e.get('field')} is {e.get('code')}" for e in data.get("errors", [])) or "no details"
        sys.exit(f"Strava sign-in failed (HTTP {res.status_code}): {data.get('message', 'unknown error')} — {problems}")

    # The log is public on a public repo, so never print token values
    if data.get("refresh_token") and data["refresh_token"] != creds["STRAVA_REFRESH_TOKEN"]:
        print("::warning::Strava issued a new refresh token. If future runs fail to sign in, "
              "re-authorize once and update the STRAVA_REFRESH_TOKEN secret.")
    print("--> Signed in to Strava silently via refresh token.")
    return data["access_token"]


def fetch_activities(access_token):
    print("--> Fetching your run/hike history from Strava...")
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"per_page": 200, "page": 1}
    activities = []
    while True:
        for attempt in range(16):
            r = requests.get("https://www.strava.com/api/v3/athlete/activities",
                             headers=headers, params=params, timeout=30)
            if r.status_code != 429:
                break
            print(f"    Strava rate limit hit — waiting 60s (attempt {attempt + 1})...")
            time.sleep(60)
        r.raise_for_status()
        page = r.json()
        if not page:
            break
        activities.extend(page)
        params["page"] += 1
    print(f"    Retrieved {len(activities)} activities.")
    return activities


def compute_fingerprint(all_activities):
    """Changes only when something that affects the map changes, so unchanged runs don't republish."""
    h = hashlib.sha256()
    for act in sorted(all_activities, key=lambda a: a.get('id', 0)):
        if act.get('type') in ACTIVITY_TYPES:
            h.update(f"{act.get('id')}|{act.get('start_date')}|{(act.get('map') or {}).get('summary_polyline', '')}\n".encode())
    h.update(datetime.datetime.now(LOCAL_TZ).date().isoformat().encode())  # 7-day window rolls daily
    h.update(GEOJSON_PATH.read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


# =========================================================================
# PART 2: MAP ENGINE (Cell 2)
# =========================================================================

def build_map(all_activities):
    if not all_activities:
        raise ValueError("No Strava activities were returned.")

    # Create a clean DataFrame from the raw cache to force reliable chronological handling
    df_acts = pd.DataFrame(all_activities)
    df_acts['start_date_dt'] = pd.to_datetime(df_acts['start_date'])
    # FORCE SORT: Oldest first, newest at the very bottom
    df_acts = df_acts.sort_values(by='start_date_dt').reset_index(drop=True)

    print("--> Step 4: Constructing your lifetime geospatial footprint...")
    activity_lines = []
    recent_activity_lines = []
    recent_activity_dates = []  # Parallel to recent_activity_lines, used for recency ordering
    latest_activity_dt = None

    # Establish our 7-day rolling window bounds based on today's date
    today_dt = datetime.datetime.now(datetime.timezone.utc)
    seven_days_ago_dt = today_dt - datetime.timedelta(days=7)

    for idx, act in df_acts.iterrows():
        act_map = act.get('map') if isinstance(act.get('map'), dict) else {}
        if act.get('type') in ACTIVITY_TYPES and act_map.get('summary_polyline'):
            coords = polyline.decode(act_map['summary_polyline'])
            flipped_coords = [(lon, lat) for lat, lon in coords]
            if len(flipped_coords) > 1:
                line = LineString(flipped_coords)
                activity_lines.append(line)

                act_date = act['start_date_dt'].to_pydatetime()
                latest_activity_dt = act_date
                # Check if activity happened within the last 7 days
                if act_date >= seven_days_ago_dt:
                    recent_activity_lines.append(line)
                    recent_activity_dates.append(act_date)

    if not activity_lines:
        raise ValueError("No activities with valid GPS tracks were found.")

    print(f"    Isolated {len(recent_activity_lines)} active track(s) from the last 7 days.")

    # Convert tracks to GeoDataFrames in UTM zone 13N meters
    my_tracks_gdf = gpd.GeoDataFrame(geometry=activity_lines, crs="EPSG:4326").to_crs(epsg=26913)
    coverage_ribbon = unary_union(my_tracks_gdf.geometry.buffer(15))

    # Build a separate coverage ribbon exclusively for tracks from the last 7 days
    if recent_activity_lines:
        recent_tracks_gdf = gpd.GeoDataFrame(geometry=recent_activity_lines, crs="EPSG:4326").to_crs(epsg=26913)
        recent_coverage_ribbon = unary_union(recent_tracks_gdf.geometry.buffer(15))
        # Per-activity buffers paired with their dates, so each trail can be tied to the latest run that covered it
        recent_track_buffers = list(zip(recent_tracks_gdf.geometry.buffer(15), recent_activity_dates))
    else:
        recent_coverage_ribbon = None
        recent_track_buffers = []

    # Build a cumulative historic ribbon *prior* to the last 7 days to see what is truly "NEW"
    historical_lines = activity_lines[:-len(recent_activity_lines)] if recent_activity_lines else activity_lines
    if historical_lines:
        historical_tracks_gdf = gpd.GeoDataFrame(geometry=historical_lines, crs="EPSG:4326").to_crs(epsg=26913)
        historical_coverage_ribbon = unary_union(historical_tracks_gdf.geometry.buffer(15))
    else:
        historical_coverage_ribbon = None

    # --- STEP B: LOAD BOULDER OSMP GEOJSON MAP DATA (from the repo) ---
    print("\n--> Step 5: Loading Boulder OSMP trail file from the repo...")
    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(f"Could not find {GEOJSON_PATH.name} in the repo.")
    osmp_gdf = gpd.read_file(GEOJSON_PATH)

    name_col = None
    for col in osmp_gdf.columns:
        if 'TRAILNAME' in col.upper() or 'TRAIL_NAME' in col.upper():
            name_col = col
            break
    if not name_col:
        text_cols = [c for c in osmp_gdf.columns if c != 'geometry' and osmp_gdf[c].dtype == 'object']
        name_col = max(text_cols, key=lambda c: osmp_gdf[c].nunique()) if text_cols else osmp_gdf.columns[0]

    # --- SPLIT DISCONNECTED SAME-NAME SEGMENTS INTO SEPARATE TRAILS ---
    # Multi-part features become individual lines so every piece is drawn and colored
    osmp_gdf = osmp_gdf.explode(index_parts=False).reset_index(drop=True)
    osmp_gdf['_name_clean'] = osmp_gdf[name_col].astype(str).str.replace('\xa0', ' ').str.strip()
    osmp_gdf['trail_key'] = osmp_gdf['_name_clean']
    _tmp_meters = osmp_gdf.to_crs(epsg=26913)

    # Segments of one trail touch at their ends; same-named pieces more than ~10 m apart are treated as separate trails
    for _nm, _grp in _tmp_meters.groupby(osmp_gdf['_name_clean']):
        if len(_grp) < 2:
            continue
        _clusters = unary_union(_grp.geometry.buffer(5))
        _parts = list(_clusters.geoms) if _clusters.geom_type == 'MultiPolygon' else [_clusters]
        if len(_parts) < 2:
            continue
        _parts.sort(key=lambda p: p.centroid.x)  # number pieces west to east for stable labels
        for _i, _geom in _grp.geometry.items():
            for _n, _part in enumerate(_parts, 1):
                if _part.intersects(_geom):
                    osmp_gdf.at[_i, 'trail_key'] = f"{_nm} ({_n})"
                    break

    osmp_gdf_meters = osmp_gdf.to_crs(epsg=26913)

    # --- BASEMAPS ---
    # USGS Topo is the default; the others use show=False so they stay off until picked in the layer switcher.
    m = folium.Map(location=[39.98, -105.26], zoom_start=12, tiles=None)

    folium.TileLayer(
        tiles="https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}",
        attr="USGS The National Map",
        name="USGS Topo",
        max_native_zoom=16,
        max_zoom=19,
        show=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri World Topo",
        max_zoom=19,
        show=False,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics",
        name="Esri Satellite",
        max_zoom=19,
        show=False,
    ).add_to(m)

    folium.TileLayer("CartoDB positron", name="CARTO Light", show=False).add_to(m)

    # --- STEP C: INTERSECTION & DASHBOARD ANALYSIS ---
    print("--> Step 6: Engineering intersection metrics and building dashboard fields...")
    trail_summary = {}

    for idx, row in osmp_gdf.iterrows():
        trail_name = row.get(name_col)
        trail_geom_wgs = row['geometry']
        trail_geom_meters = osmp_gdf_meters.loc[idx, 'geometry']

        if trail_name and trail_geom_meters and trail_geom_meters.length > 0:
            trail_name_clean = row['trail_key']

            # Calculate intersections
            intersection_lifetime = trail_geom_meters.intersection(coverage_ribbon)
            intersection_recent = trail_geom_meters.intersection(recent_coverage_ribbon) if recent_coverage_ribbon is not None else None
            intersection_historic = trail_geom_meters.intersection(historical_coverage_ribbon) if historical_coverage_ribbon is not None else None

            tot_miles = trail_geom_meters.length * 0.000621371
            cov_miles_lifetime = intersection_lifetime.length * 0.000621371
            cov_miles_recent = intersection_recent.length * 0.000621371 if intersection_recent else 0.0
            cov_miles_historic = intersection_historic.length * 0.000621371 if intersection_historic else 0.0

            if trail_name_clean not in trail_summary:
                trail_summary[trail_name_clean] = {
                    'total_miles': 0.0,
                    'covered_miles_lifetime': 0.0,
                    'covered_miles_recent': 0.0,
                    'covered_miles_historic': 0.0,
                    'latest_recent_date': None
                }

            trail_summary[trail_name_clean]['total_miles'] += tot_miles
            trail_summary[trail_name_clean]['covered_miles_lifetime'] += cov_miles_lifetime
            trail_summary[trail_name_clean]['covered_miles_recent'] += cov_miles_recent
            trail_summary[trail_name_clean]['covered_miles_historic'] += cov_miles_historic

            # Record the most recent activity in the last 7 days that ran a meaningful stretch of this segment
            # (threshold ignores tracks that merely cross the trail)
            min_overlap_m = min(50.0, 0.5 * trail_geom_meters.length)
            for track_buffer, track_date in recent_track_buffers:
                if trail_geom_meters.intersection(track_buffer).length >= min_overlap_m:
                    prev = trail_summary[trail_name_clean]['latest_recent_date']
                    if prev is None or track_date > prev:
                        trail_summary[trail_name_clean]['latest_recent_date'] = track_date

            # Draw map polylines
            pct_seg = (intersection_lifetime.length / trail_geom_meters.length) * 100
            color = 'green' if pct_seg >= 90.0 else ('orange' if pct_seg > 5.0 else 'red')

            if trail_geom_wgs and trail_geom_wgs.geom_type == 'LineString':
                sim_geom = trail_geom_wgs.simplify(0.0001)
                coords = [(lat, lon) for lon, lat in sim_geom.coords]
                popup_text = f"<b>{trail_name_clean}</b><br>Segment Length: {tot_miles:.2f} mi"
                folium.PolyLine(coords, color=color, weight=3, opacity=0.8, popup=popup_text).add_to(m)

    # --- STEP D: SCALING METRIC STATS ---
    total_osmp_trails = len(trail_summary)
    total_system_network_miles = 0.0
    lifetime_completed_count = 0
    lifetime_miles_completed = 0.0

    recent_new_completed_count = 0
    recent_repeat_completed_count = 0
    recent_footprint_miles = 0.0
    recent_new_footprint_miles = 0.0

    weekly_active_completed_trail_records = []
    report_rows = []

    for t_name, data in sorted(trail_summary.items()):
        tot_mi = data['total_miles']
        cov_mi_life = data['covered_miles_lifetime']
        cov_mi_rec = data['covered_miles_recent']
        cov_mi_hist = data['covered_miles_historic']

        total_system_network_miles += tot_mi

        pct_life = (cov_mi_life / tot_mi) * 100 if tot_mi > 0 else 0.0
        pct_hist = (cov_mi_hist / tot_mi) * 100 if tot_mi > 0 else 0.0
        pct_rec = (cov_mi_rec / tot_mi) * 100 if tot_mi > 0 else 0.0

        # Track Lifetime Completed
        if pct_life >= 90.0:
            lifetime_completed_count += 1
            status = "YES (100%)"
        elif pct_life > 5.0:
            status = f"In Progress ({pct_life:.1f}%)"
        else:
            status = "NO"

        lifetime_miles_completed += cov_mi_life

        # Track 7-Day Rolling Dynamics cleanly
        is_recent_completion = (pct_life >= 90.0 and pct_hist < 90.0)
        is_strict_repeat_completion = (pct_life >= 90.0 and pct_hist >= 90.0 and pct_rec >= 90.0)

        if is_recent_completion:
            recent_new_completed_count += 1
            weekly_active_completed_trail_records.append({'name': t_name, 'type': 'NEW', 'date': data['latest_recent_date']})
        elif is_strict_repeat_completion:
            recent_repeat_completed_count += 1
            weekly_active_completed_trail_records.append({'name': t_name, 'type': 'REPEAT', 'date': data['latest_recent_date']})

        # Option B Mileage Math
        recent_footprint_miles += cov_mi_rec
        new_miles_this_week = max(0.0, cov_mi_life - cov_mi_hist)
        recent_new_footprint_miles += new_miles_this_week

        report_rows.append({
            'Trail Name': t_name,
            'Completed': status,
            'Total Segment Miles': round(tot_mi, 2),
            'Miles Remaining': round(max(0.0, tot_mi - cov_mi_life), 2),
            'Map Resource Link': "https://bouldercolorado.gov/services/open-space-and-mountain-parks-trails"
        })

    # Total Completions = New Breakthroughs + Strict End-to-End Repeats
    total_weekly_completions_count = recent_new_completed_count + recent_repeat_completed_count

    # Formatted whole miles values for display
    display_miles_completed = int(round(lifetime_miles_completed))
    display_total_system_miles = int(round(total_system_network_miles))
    latest_activity_label = latest_activity_dt.astimezone(LOCAL_TZ).strftime("%a %b %-d, %-I:%M %p")

    # Build updated color-coded HTML list elements string
    if weekly_active_completed_trail_records:
        # Most recent first; trails finished on the same activity stay alphabetical
        oldest_possible = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
        sorted_records = sorted(weekly_active_completed_trail_records, key=lambda x: x['name'])
        sorted_records = sorted(sorted_records, key=lambda x: x['date'] or oldest_possible, reverse=True)

        list_items = ""
        for rec in sorted_records:
            if rec['type'] == 'NEW':
                list_items += f"<li style='margin-bottom: 5px; color: #6ee7b7;'><span style='font-weight: 800; font-size: 8px; background-color: rgba(110,231,183,0.15); padding: 2px 4px; border-radius: 3px; margin-right: 6px;'>NEW</span> {rec['name']}</li>"
            else:
                list_items += f"<li style='margin-bottom: 4px; color: #7dd3fc;'><span style='font-weight: 800; font-size: 8px; background-color: rgba(125,211,252,0.15); padding: 2px 4px; border-radius: 3px; margin-right: 6px;'>REPEAT</span> {rec['name']}</li>"

        weekly_list_html = f"""
        <div style='margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.08); font-size: 11px;'>
            <div style='font-weight: 700; color: #a3b8cc; text-transform: uppercase; font-size: 9px; letter-spacing: 0.5px; margin-bottom: 6px;'>Completed This Week:</div>
            <ul class='dash-list' style='margin: 0; padding-left: 2px; max-height: 480px; overflow-y: auto; list-style-type: none;'>
                {list_items}
            </ul>
        </div>
        """
    else:
        weekly_list_html = """
        <div style='margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.08); font-size: 11px; color: #aaa; font-style: italic;'>
            No trail completions logged yet this week.
        </div>
        """

    # --- STEP E: CUSTOM INJECTED DASHBOARD UI PANEL ---
    class FloatingDashboard(MacroElement):
        def __init__(self, html_content):
            super(FloatingDashboard, self).__init__()
            self._template = Template(f"""
                {{% macro script(this, kwargs) %}}
                $('body').append(`{html_content}`);
                if (window.innerWidth < 600) {{ document.getElementById('map-dashboard').classList.add('collapsed'); }}
                {{% endmacro %}}
            """)

    # Phone layout: narrower panel, tap the title to collapse/expand (starts collapsed on phones)
    dashboard_css = """
    <style>
        #map-dashboard h3 { cursor: pointer; user-select: none; }
        #map-dashboard .dash-caret { float: right; transition: transform 0.2s; }
        #map-dashboard.collapsed .dash-body { display: none; }
        #map-dashboard.collapsed h3 { margin-bottom: 0 !important; border-bottom: none !important; padding-bottom: 0 !important; }
        #map-dashboard.collapsed .dash-caret { transform: rotate(-90deg); }
        @media (max-width: 600px) {
            #map-dashboard {
                top: 10px !important; right: 10px !important; padding: 12px !important;
                width: calc(100vw - 80px) !important; max-width: 340px; box-sizing: border-box;
                max-height: calc(100vh - 20px); overflow-y: auto;
            }
            #map-dashboard .dash-list { max-height: 40vh !important; }
        }
    </style>
    """

    dashboard_html = f"""
    {dashboard_css}
    <div id="map-dashboard" style="
        position: fixed;
        top: 20px;
        right: 20px;
        width: 340px;
        background-color: rgba(30, 34, 42, 0.9);
        color: #e6ebf5;
        z-index: 9999;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(6px);
    ">
        <h3 onclick="document.getElementById('map-dashboard').classList.toggle('collapsed')" style="margin-top: 0; margin-bottom: 12px; font-size: 15px; border-bottom: 1px solid rgba(255,255,255,0.15); padding-bottom: 6px; color: #52a3ff; font-weight: 600; letter-spacing: 0.5px;">
            🗺️ Boulder OSMP Trail Tracker <span class="dash-caret">▾</span>
        </h3>

        <div class="dash-body">
            <div style="font-size: 12px; margin-bottom: 14px; line-height: 1.5;">
                <div style="font-weight: 700; color: #9aa5b5; text-transform: uppercase; font-size: 9px; letter-spacing: 0.8px; margin-bottom: 4px;">Lifetime Statistics</div>
                <div style="display: flex; justify-content: space-between;"><span>Trails Completed:</span> <strong style="color: #ffffff;">{lifetime_completed_count} of {total_osmp_trails}</strong></div>
                <div style="display: flex; justify-content: space-between;"><span>Footprint Distance:</span> <strong style="color: #ffffff;">{display_miles_completed} of {display_total_system_miles} mi</strong></div>
                <div style="display: flex; justify-content: space-between;"><span>Latest Activity:</span> <strong style="color: #ffffff;">{latest_activity_label}</strong></div>
            </div>

            <div style="font-size: 12px; background-color: rgba(255,255,255,0.04); padding: 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.04); line-height: 1.5;">
                <div style="font-weight: 700; color: #ff7e5f; text-transform: uppercase; font-size: 9px; letter-spacing: 0.8px; margin-bottom: 6px; display: flex; align-items: center;">
                    ⚡ Last 7 Days
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 3px;"><span>OSMP Trails Completed:</span> <strong style="color: #ffffff;">{total_weekly_completions_count}</strong></div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 3px;"><span>New OSMP Trails Completed:</span> <strong style="color: #6ee7b7;">{recent_new_completed_count}</strong></div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 3px;"><span>OSMP Mileage Footprint:</span> <strong style="color: #ffffff;">{recent_footprint_miles:.2f} mi</strong></div>
                <div style="display: flex; justify-content: space-between;"><span>New OSMP Distance added:</span> <strong style="color: #6ee7b7;">+{recent_new_footprint_miles:.2f} mi</strong></div>

                {weekly_list_html}
            </div>
        </div>
    </div>
    """

    m.add_child(FloatingDashboard(dashboard_html))
    # Layer switcher sits top-left so it isn't hidden behind the dashboard panel
    m.add_child(folium.LayerControl(position='topleft', collapsed=True))
    # "Show my location" button (works because GitHub Pages serves the map over HTTPS)
    LocateControl(
        position='topleft',
        flyTo=True,
        strings={'title': 'Show my location'},
        locateOptions={'enableHighAccuracy': True, 'maxZoom': 16},
    ).add_to(m)
    m.get_root().title = "Boulder OSMP Trail Tracker"

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / ".nojekyll").touch()  # tells GitHub Pages to serve files as-is
    m.save(str(MAP_PATH))
    pd.DataFrame(report_rows).to_csv(CSV_PATH, index=False)

    print("\n=======================================================")
    print(f"🎉 MAP PUBLISHED: {lifetime_completed_count} of {total_osmp_trails} trails complete")
    print("=======================================================")


def main():
    access_token = get_access_token()
    all_activities = fetch_activities(access_token)

    fingerprint = compute_fingerprint(all_activities)
    if FINGERPRINT_PATH.exists() and FINGERPRINT_PATH.read_text().strip() == fingerprint:
        print("--> No new activities or trail-data changes since the last build. Map left as-is.")
        return

    build_map(all_activities)
    FINGERPRINT_PATH.write_text(fingerprint + "\n")


if __name__ == "__main__":
    main()
