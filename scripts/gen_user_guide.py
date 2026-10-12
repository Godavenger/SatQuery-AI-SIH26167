"""SatQuery AI (SIH26167) - Website User Guide PDF.
A step-by-step walkthrough of the Earth-view web app (docs/index.html),
matching the real UI: header, search, missions/imagery panels, pass view,
workspace (dossier / ask-AI / scenes), time lapse, insights, shortcuts.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdf_style_kit import (install_fonts, styles, section, h2, make_table,
                           kpi_row, MARGINS, ACCENT, TEXT_MUTED)
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'docs', 'SatQuery_AI_User_Guide.pdf')
S = styles()

STEP_COLORS = None  # single-accent design; steps numbered textually


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont('Calibri', 7.5)
    canvas.setFillColor(TEXT_MUTED)
    canvas.drawString(doc.leftMargin, A4[1] - 1.15 * cm,
                      'SatQuery AI  ·  SIH26167  ·  Website User Guide')
    canvas.drawRightString(A4[0] - doc.rightMargin, A4[1] - 1.15 * cm,
                           '12 September 2026')
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(1.2)
    canvas.line(doc.leftMargin, A4[1] - 1.35 * cm,
                A4[0] - doc.rightMargin, A4[1] - 1.35 * cm)
    canvas.line(doc.leftMargin, 1.25 * cm, A4[0] - doc.rightMargin, 1.25 * cm)
    canvas.setFont('Calibri', 7.5)
    canvas.drawString(doc.leftMargin, 0.95 * cm,
                      'Team SatQuery AI  ·  Smart India Hackathon 2026')
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.95 * cm,
                          'Page %d' % doc.page)
    canvas.restoreState()


def P(txt, st='body'):
    return Paragraph(txt, S[st])


def K(keys):
    """Render keyboard keys in monospace chips."""
    return ' '.join('<font name="Consolas" color="#b5273e">%s</font>'
                    % k for k in keys)


def main():
    install_fonts()
    doc = SimpleDocTemplate(OUT, pagesize=A4, title='SatQuery AI Website '
                           'User Guide (SIH26167)', author='Z.ai',
                           creator='Z.ai', subject='Step-by-step guide to '
                           'using the SatQuery AI Earth-view web application',
                           **MARGINS)
    story = []

    # ---------- title ----------
    story.append(Spacer(1, 4))
    story.append(P('<b>SatQuery AI</b>', 'title'))
    story.append(Spacer(1, 2))
    story.append(P('Website User Guide — Earth View & Analysis '
                   'Workspace', 'subtitle'))
    story.append(Spacer(1, 6))
    story.append(P('Smart India Hackathon 2026 · Problem SIH26167 '
                   '(ISRO / Space Technology)   |   '
                   'Report date: 12 September 2026', 'meta'))
    story.append(Spacer(1, 10))
    story.append(P(
        'This guide walks through the SatQuery AI web application in the '
        'order you would actually use it: launch, read the screen, navigate '
        'the globe, explore imagery layers, open mission sites, analyse a '
        'region in the workspace, ask the AI questions, play time lapses, '
        'and export session results. Follow it top to bottom for a first '
        'tour, or jump to any numbered section.'))

    # ---------- 1. getting started ----------
    story += section(S, 1, 'Getting started')
    story += h2('Launching the app', S)
    launch = [[P('<b>Option</b>', 'th'), P('<b>Steps</b>', 'th')]]
    launch.append([P('One-click (recommended)'),
                   P('Double-click <font name="Consolas">START_DEMO.bat</font> '
                     'in the project root. It starts the AI backend on port '
                     '7860 and opens the app in your default browser.')])
    launch.append([P('Manual'),
                   P('Start the backend: <font name="Consolas">python -m '
                     'uvicorn satquery.api:app --port 7860</font>, then open '
                     '<font name="Consolas">docs/index.html</font> through a '
                     'local server or file URL.')])
    launch.append([P('View-only'),
                   P('Just open <font name="Consolas">docs/index.html</font> '
                     'in any browser. The globe, imagery and dossiers work '
                     'without the backend; only live AI answers need it.')])
    story.append(make_table(launch, [0.24, 0.76]))
    story.append(Spacer(1, 6))
    story.append(P(
        'On load you see a brief splash, then the live Earth globe. The '
        'header line shows which satellite feed is active and the date it '
        'was captured, e.g. VIIRS SNPP, captured 2026-09-08 (latest '
        'available). That suffix means imagery over some regions stepped '
        'back a day because NASA had not finished ingesting today’s '
        'passes there — this is intentional and guarantees you never see '
        'blacked-out hemispheres. The globe needs an internet connection '
        '(NASA GIBS + Esri imagery are streamed live).'))

    # ---------- 2. reading the screen ----------
    story += section(S, 2, 'Reading the screen')
    scr = [[P('<b>Region</b>', 'th'), P('<b>What it is</b>', 'th')]]
    scr += [
        (P('Header (top-left)'),
         P('SatQuery AI title, the active satellite feed + captured date, '
           'a GIBS LIVE badge, and three buttons: Insights, Docs & API, '
           'and Keyboard shortcuts.')),
        (P('Search pill (top-centre)'),
         P('One box that finds places, mission sites, or raw coordinates '
           'like <font name="Consolas">26.14, 91.73</font>. Press Enter to '
           'fly there; arrow keys pick a suggestion.')),
        (P('Missions panel (left)'),
         P('Category chips (All, Flood, Urban, Coastal, Agriculture, '
           'Forest, Wildfire, Glacier) and the nine mission-site cards. '
           'Click a card to preview; it opens the site dossier popup.')),
        (P('Imagery panel (right)'),
         P('The eight base imagery layers (keys 1–8), the analytic '
           'overlay stack with an “+ Add overlay” menu, and '
           'context toggles (borders, place labels, graticule, live '
           'activity).')),
        (P('Status bar (bottom)'),
         P('Cursor lat/lon, current view, session counters (questions '
           'asked, sites analysed, 5-capability coverage) and frame rate.')),
    ]
    story.append(make_table(scr, [0.24, 0.76]))
    story.append(Spacer(1, 4))
    story.append(P('Table 1 — The five permanent screen regions.',
                   'caption'))

    # ---------- 3. navigating ----------
    story += section(S, 3, 'Navigating the globe')
    story.append(P(
        'Navigation follows three levels of detail, each with its own '
        'imagery source:'))
    lvl = [[P('<b>Level</b>', 'th'), P('<b>You see</b>', 'th'),
            P('<b>How to get there</b>', 'th')]]
    lvl += [
        (P('<b>1 — Globe</b>'),
         P('Whole-Earth view, NASA daily satellite mosaics (VIIRS / MODIS, '
           '~250–375 m per pixel).'),
         P('Drag to orbit. Scroll to zoom toward the cursor.')),
        (P('<b>2 — Pass view</b>'),
         P('A flat high-resolution regional pass of wherever you zoomed.'),
         P('Keep zooming in — the viewer hands off automatically '
           'around a few hundred km. ' + K(['Shift', '+', 'click']) +
           ' inside it for even higher resolution.')),
        (P('<b>3 — Aerial</b>'),
         P('Sub-metre aerial imagery (Esri) at street / building scale.'),
         P('Zoom under roughly 60 km width; the handoff is seamless and '
           'automatic.')),
    ]
    story.append(make_table(lvl, [0.18, 0.42, 0.40]))
    story.append(Spacer(1, 4))
    story.append(P('Table 2 — The three zoom levels.', 'caption'))
    story.append(Spacer(1, 2))
    story.append(P(
        'Hold ' + K(['Shift']) + ' while scrolling for a strong 3× '
        'zoom. Press ' + K(['Esc']) + ' anywhere to back out one level '
        '(aerial → pass → globe → close dialogs). ' +
        K(['Shift', '+', 'click']) + ' or right-click any point on Earth '
        'to jump straight to the analysis workspace at that location.'))

    # ---------- 4. finding places ----------
    story += section(S, 4, 'Finding a place')
    story.append(P(
        'Click the search pill and start typing. It matches three things: '
        'place names, the nine mission sites, and raw coordinates. For '
        'example typing <font name="Consolas">assam</font> or '
        '<font name="Consolas">26.14, 91.73</font> both take you to the '
        'Assam flood plain. Use the up/down arrows to walk the suggestion '
        'list and Enter to fly; Esc dismisses it. The globe animates to the '
        'selected location.'))

    # ---------- 5. imagery layers ----------
    story += section(S, 5, 'Choosing imagery')
    story += h2('Base layers (keys 1–8)', S)
    base = [[P('<b>Key</b>', 'th'), P('<b>Layer</b>', 'th'),
             P('<b>Best for</b>', 'th')]]
    base += [
        (P('1', 'tdc'), P('VIIRS SNPP true color'),
         P('Default. Daily global natural-color imagery; storms, smoke, '
           'snow, sea ice.')),
        (P('2', 'tdc'), P('VIIRS NOAA-20 true color'),
         P('Second daily pass when SNPP is cloudy or late.')),
        (P('3', 'tdc'), P('MODIS Terra true color'),
         P('Long-archive morning pass; same-day VIIRS fills equatorial '
           'gaps.')),
        (P('4', 'tdc'), P('MODIS Aqua true color'),
         P('Afternoon pass; compare cloud cover against Terra.')),
        (P('5', 'tdc'), P('VIIRS SWIR false color'),
         P('Burn scars, floods, snow/ice vs cloud separation.')),
        (P('6', 'tdc'), P('VIIRS cloud/snow false color'),
         P('Separating snow and ice from cloud.')),
        (P('7', 'tdc'), P('Black Marble night lights'),
         P('City footprints and infrastructure at night (2016 composite).')),
        (P('8', 'tdc'), P('VIIRS N20 SWIR false color'),
         P('Fire and flood contrast on the alternate VIIRS satellite.')),
    ]
    story.append(make_table(base, [0.08, 0.32, 0.60]))
    story.append(Spacer(1, 4))
    story.append(P('Table 3 — Base imagery layers, switchable with '
                   'number keys 1–8 or the Imagery panel.', 'caption'))
    story.append(Spacer(1, 2))

    story += h2('Overlays and context', S)
    story.append(P(
        'Click “+ Add overlay” in the Imagery panel to stack '
        'translucent analytic layers on top of the base: aerosol optical '
        'depth (smoke/dust plumes), cloud-top temperature, precipitable '
        'water, sea-surface temperature, chlorophyll-a, snow cover, sea '
        'ice, and active-fire detections from VIIRS and MODIS. Overlays '
        'are meant to be read against a true-color base — e.g. fire '
        'dots over smoke. The Context toggles add reference geography: '
        'country borders (' + K(['B']) + '), place labels for 76,000+ '
        'cities, towns and villages (' + K(['L']) + '), the lat/lon '
        'graticule (' + K(['G']) + '), and live event markers from USGS '
        'earthquakes and NASA EONET (' + K(['A']) + ').'))

    # ---------- 6. mission sites ----------
    story += section(S, 6, 'Mission sites and dossiers')
    story.append(P(
        'The nine mission sites are curated analysis locations: Assam '
        'Flood Plain, Delhi Urban Sprawl, Sundarbans Delta, Punjab Agri '
        'Belt, Gangotri Glacier, Rondônia Deforestation, Sierra '
        'Nevada Fire Scar, Nile Delta, and Dubai Coast. Filter them with '
        'the category chips. Clicking a site card flies the globe there '
        'and opens a popup with its story; from the popup, <b>Open '
        'workspace</b> enters the full analysis workspace and '
        '<b>Refocus</b> just re-centres the view. The site markers on the '
        'globe itself are also clickable.'))

    # ---------- 7. workspace ----------
    story += section(S, 7, 'The analysis workspace')
    story.append(P(
        'Open the workspace in any of three ways: click a mission site and '
        'choose <b>Open workspace</b>; ' + K(['Shift', '+', 'click']) +
        ' anywhere on Earth; or right-click a point. It is a large window '
        'with the imagery viewer on the left and three tabs on the right.'))
    ws = [[P('<b>Control</b>', 'th'), P('<b>What it does</b>', 'th')]]
    ws += [
        (P('Date picker'),
         P('Choose which day’s imagery to display (steps back '
           'automatically when the newest day has no coverage there).')),
        (P('Layer selector'),
         P('Swap the imagery layer for this window independently of the '
           'globe.')),
        (P('Time lapse'),
         P('Opens the playback modal (section 9).')),
        (P('Info'),
         P('Shows imagery metadata: source, resolution, date, coverage.')),
        (P('Download frame'),
         P('Saves the current imagery frame as an image file.')),
        (P('Google Maps'),
         P('Opens the same coordinates in Google Maps for cross-reference.')),
        (P('Viewer canvas'),
         P('Drag to pan, scroll to zoom, shift-click to recenter the '
           'window on a point.')),
    ]
    story.append(make_table(ws, [0.24, 0.76]))
    story.append(Spacer(1, 4))
    story.append(P('Table 4 — Workspace viewer controls.', 'caption'))
    story.append(Spacer(1, 2))
    story.append(P(
        'The footer under the viewer shows coordinates, window span, '
        'resolution and imagery date; a scale bar sits at the right. '
        'Answers from the AI draw gold dashed boxes directly on this '
        'canvas.'))

    story += h2('The three tabs', S)
    tabs = [[P('<b>Tab</b>', 'th'), P('<b>Contents</b>', 'th')]]
    tabs += [
        (P('<b>Dossier</b>'),
         P('The site’s analysis brief: the problem, approach, KPIs, '
           'an event timeline, the sensors used, findings with confidence '
           'values, and recommended actions.')),
        (P('<b>Ask AI</b>'),
         P('The question box and the agent pipeline (section 8).')),
        (P('<b>Scenes</b>'),
         P('The raw Sentinel-2 / Sentinel-1 scene metadata behind the '
           'analysis pair.')),
    ]
    story.append(make_table(tabs, [0.18, 0.82]))
    story.append(Spacer(1, 4))
    story.append(P('Table 5 — Workspace side tabs.', 'caption'))

    # ---------- 8. asking the AI ----------
    story += section(S, 8, 'Asking the AI')
    story.append(P(
        'In the workspace, open the <b>Ask AI</b> tab. Type a question '
        'about the current window and press Ask (or Enter). Five preset '
        'buttons Q1–Q5 cover the canonical query types. As the query '
        'runs you watch the six pipeline stages animate — register, '
        'fetch, featurise, reason, ground, report — then the answer '
        'appears with a confidence value, and any region it refers to is '
        'boxed on the imagery. A <b>trace JSON</b> button exposes the '
        'full audit record, and the model registry shows which head went '
        'live for your query.'))
    q = [[P('<b>Ask about…</b>', 'th'),
          P('<b>Example phrasing</b>', 'th'),
          P('<b>Head that answers</b>', 'th')]]
    q += [
        (P('What is here'), P('Describe the land cover and major '
          'objects visible in this image.'), P('Caption', 'tdc')),
        (P('Where something is'),
         P('Highlight the water body referred to in the query.'),
         P('Grounding', 'tdc')),
        (P('What changed'),
         P('What changed between these two dates, and where?'),
         P('Change', 'tdc')),
        (P('Change Q&A'),
         P('Has the built-up area increased, decreased, or stayed '
           'the same?'), P('Change-qa', 'tdc')),
        (P('Sensor fusion'),
         P('Use the optical and SAR images together to identify '
           'built-up and water regions.'), P('Fusion', 'tdc')),
        (P('Anything else'),
         P('Free-form questions about the window.'), P('VQA', 'tdc')),
    ]
    story.append(make_table(q, [0.20, 0.60, 0.20]))
    story.append(Spacer(1, 4))
    story.append(P('Table 6 — Query types and the head each one '
                   'activates.', 'caption'))
    story.append(Spacer(1, 2))
    story.append(P(
        'The app runs in two modes. With the backend running it answers '
        'from the real agent pipeline (the first query after a cold start '
        'is slow while the model loads; later ones are under a second). '
        'Without the backend it answers in demonstration mode with the '
        'same interface, so the tour never breaks — the header '
        'badge and answer text tell you which mode you are in.'))

    # ---------- 9. time lapse ----------
    story += section(S, 9, 'Time lapse')
    story.append(P(
        'From any workspace, click <b>Time lapse</b>. Pick 7 or 30 days, '
        'then use Play / Prev / Next or the slider to step through daily '
        'frames of the same window. The date of the current frame is '
        'overlaid on the imagery. Use it to watch floods rise, fires '
        'spread, snow retreat or a city expand.'))

    # ---------- 10. insights ----------
    story += section(S, 10, 'Insights and exports')
    story.append(P(
        'The Insights button in the header (or ' + K(['I']) + ') opens '
        'session telemetry: every question you asked, which sites and '
        'capabilities were covered out of the five prescribed ones, and '
        'timing. From here you can export the whole session as JSON or '
        'CSV, or reset it. The Docs & API button (' + K(['D']) +
        ') documents the agent pipeline, the model registry and the data '
        'sources behind the app.'))

    # ---------- 11. shortcuts ----------
    story += section(S, 11, 'Keyboard shortcuts')
    sc = [[P('<b>Keys</b>', 'th'), P('<b>Action</b>', 'th')]]
    sc += [
        (P(K(['1']) + ' – ' + K(['8'])), P('Switch base imagery layer')),
        (P(K(['B'])), P('Toggle country borders')),
        (P(K(['L'])), P('Toggle place labels (76,000+ places)')),
        (P(K(['G'])), P('Toggle lat/lon graticule')),
        (P(K(['A'])), P('Toggle live activity markers (USGS / EONET)')),
        (P(K(['Shift', '+', 'scroll'])), P('Strong zoom')),
        (P(K(['Shift', '+', 'click'])), P('Open workspace at that point')),
        (P(K(['right-click'])), P('Open workspace at that point')),
        (P(K(['drag'])), P('Orbit the globe / pan imagery')),
        (P(K(['scroll'])), P('Cursor-anchored zoom')),
        (P(K(['Esc'])), P('Close dialogs / back out one zoom level')),
        (P(K(['I'])), P('Insights')),
        (P(K(['D'])), P('Docs & API')),
        (P(K(['?'])), P('Keyboard shortcuts panel')),
    ]
    story.append(make_table(sc, [0.30, 0.70]))
    story.append(Spacer(1, 4))
    story.append(P('Table 7 — Complete shortcut list (also available '
                   'in-app via the ? panel).', 'caption'))

    # ---------- 12. tips ----------
    story += section(S, 12, 'Tips and troubleshooting')
    tips = [[P('<b>Symptom / need</b>', 'th'), P('<b>What to do</b>', 'th')]]
    tips += [
        (P('Header shows an older date with the suffix '
           '(latest available)'),
         P('Expected. NASA ingests daily passes region-by-region over '
           '~20 hours; the app automatically steps back to the newest '
           'date with full coverage rather than showing black gaps.')),
        (P('Black or missing imagery in a region'),
         P('Try another base layer (key 2 or 3) — different '
           'satellites ingest at different times. Also check your '
           'internet connection; all imagery is streamed live.')),
        (P('First AI answer is slow'),
         P('The backend lazy-loads its model on the first query '
           '(~340 MB). Subsequent queries answer in under a second.')),
        (P('AI answers look canned'),
         P('The backend is not running. Start it with START_DEMO.bat '
           'and reload the page; the app auto-detects it within a '
           'couple of seconds.')),
        (P('Water questions abstain on some sites'),
         P('By design. If there is no water in the window (e.g. the dry '
           'Assam upland patch), the system abstains rather than '
           'guessing. Try Kaziranga or the Sundarbans for water '
           'queries.')),
        (P('A dialog will not close'),
         P('Press ' + K(['Esc']) + '. It closes the topmost dialog, or '
           'backs out of the pass view.')),
    ]
    story.append(make_table(tips, [0.34, 0.66]))
    story.append(Spacer(1, 4))
    story.append(P('Table 8 — Quick troubleshooting.', 'caption'))

    story.append(Spacer(1, 10))
    story.append(P(
        'Suggested five-minute tour: open the app → press 5 for SWIR '
        '→ click the Assam Flood Plain card → Open workspace '
        '→ press Q3 (change question) → watch the boxes land on '
        'the imagery → click Time lapse, 30 days, Play → close '
        'with Esc → press I to see your session recorded.'))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
