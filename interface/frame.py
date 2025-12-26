from contextlib import contextmanager

from nicegui import app, ui

app.add_static_files('/static', 'static')
app.add_static_files('/fonts', 'fonts')

@contextmanager
def frame(navigation_title: str):   #, menu: callable
    """Custom page frame to share the same styling and behavior across all pages"""
    #ui.colors(primary='#000000', secondary='#000000', accent='#111B1E', positive='#53B689')
    ui.add_head_html(f'<link href="/static/styles.css" rel="stylesheet" type="text/css">')
    with ui.header().style('background-color: #ffffff').classes("flex items-center justify-center"):
        ui.image("/static/isw.png").classes("w-32")
        ui.label('AAS-graph-comparison').style('color: #000000;')
        ui.space()
        ui.label(navigation_title).style('color: #000000;')
        #menu()
    with ui.column().classes('w-full'):
        yield