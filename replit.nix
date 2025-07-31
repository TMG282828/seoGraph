{ pkgs }: {
  deps = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.setuptools
    pkgs.python311Packages.wheel
    pkgs.python311Packages.virtualenv
    pkgs.python311Packages.python-lsp-server
    pkgs.postgresql
    pkgs.sqlite
    pkgs.git
  ];
  env = {
    PYTHON_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
      pkgs.glib
      pkgs.xorg.libX11
      pkgs.glibc
    ];
    PYTHONPATH = "$PWD";
    PYTHONHASHSEED = "0";
    MPLBACKEND = "Agg";
    PYTHONUNBUFFERED = "1";
    PYTHONDONTWRITEBYTECODE = "1";
    PIP_DISABLE_PIP_VERSION_CHECK = "1";
    PIP_NO_CACHE_DIR = "1";
    REPL_LANGUAGE = "python3";
  };
}