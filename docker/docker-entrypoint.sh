#! /bin/bash
mkdir -p /logs/query-rewrite/$HOSTNAME/logs

/usr/bin/supervisord -c /app/query-rewrite/docker/supervisord.conf

tail -f /dev/null