#!/bin/bash

rm -f _r2sol.so
swig -c++ -python r2sol.i
python setup.py build_ext
cp -f build/lib.linux-x86_64-3.7/_r2sol.cpython-37m-x86_64-linux-gnu.so ./_r2sol.so
rm -f -r build

if [ "$1" == "local" ]; then
  python run_local.py
else
  python run.py
fi

rm -f -r __pycache__
rm -f _r2sol.so

