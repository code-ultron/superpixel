
import sys

sys.path.append('/home/janischl/zampal/webserver')
sys.path.append('/workspace')

from config import Config

print("reloaded")
bind = '0.0.0.0:8886'
backlog = 2048

workers = 1
worker_class = 'eventlet'
worker_connections = 1000
timeout = 30000
keepalive = 150

reload = Config.DEBUG
preload = Config.PRELOAD

errorlog = '-'
loglevel = Config.LOG_LEVEL
accesslog = '-'