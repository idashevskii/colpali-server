#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../"

dcCommand="docker compose -f docker-compose.yml -f docker-compose.prod.yml"
$dcCommand build
$dcCommand up -d
