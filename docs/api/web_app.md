# API-Referenz — Web-App

`web_app.server` — Flask-Server, Worker-Threads und REST-API.

## Worker-Klassen

::: web_app.server.ModuleWorker
::: web_app.server.DiebWorker
::: web_app.server.WaffeWorker
::: web_app.server.GesichtWorker
::: web_app.server.FahndungWorker
::: web_app.server.TrackWorker
::: web_app.server.CameraFeed
::: web_app.server.EventLog
::: web_app.server.Runtime

## REST-API (Flask-Routen)

::: web_app.server.main
::: web_app.server.api_status
::: web_app.server.api_events
::: web_app.server.api_config_get
::: web_app.server.api_config_set
::: web_app.server.api_module_start
::: web_app.server.api_module_stop
::: web_app.server.api_ptz
::: web_app.server.api_stream
::: web_app.server.api_frame
::: web_app.server.api_faces_list
::: web_app.server.api_faces_add
::: web_app.server.api_faces_remove
::: web_app.server.api_fahndung_list
::: web_app.server.api_fahndung_update
::: web_app.server.api_fahndung_scan
::: web_app.server.api_alarm_test