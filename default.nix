with (import <nixpkgs> {});
let
  LLP = with pkgs; [
    gcc11
    cudatoolkit
    linuxPackages.nvidia_x11
    go
    cmake
  ];
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath LLP;
in  
stdenv.mkDerivation {
  name = "ollama-env";
  buildInputs = LLP;
  src = null;
  # IMPORTANT: Edit ./llm/generate/gen_linux.sh
  shellHook = ''
    SOURCE_DATE_EPOCH=$(date +%s)
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
    export NIX_CUDA_OUT_DIR=${cudatoolkit.out}
    export NIX_CUDA_LIB_DIR=${cudatoolkit.lib}
    export NVCC_PREPEND_FLAGS='-ccbin ${gcc11}/bin/'
    #export GIN_MODE=release
  '';
}
