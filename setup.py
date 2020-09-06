from distutils.core import setup, Extension
r2sol_module = Extension('_r2sol',
   sources=['r2sol_wrap.cxx', 'r2sol.cc'],
)
setup (name = 'r2sol',
   version = '0.1',
   author = "SWIG Docs",
   description = """Swig for r2sol""",
   ext_modules = [r2sol_module],
   py_modules = ["r2sol"],
)
