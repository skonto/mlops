#!/bin/bash

set -ex


if [[ "$1" == "torchserve" ]]; then
    shift
    torchserve "$@" --start
else
    exec "$@"
fi

tail -f /dev/null