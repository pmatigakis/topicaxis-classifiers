#!/bin/bash

curl -X POST -H "content-type: application/json" --data '["hello world"]' http://127.0.0.1:3000/categories/v1
curl -X POST -H "content-type: application/json" --data '["hello world"]' http://127.0.0.1:3000/categories/v2
