{
  description = "ReBoot -> ckks-dialect MLIR for KeyMemRT";

  # The forked OpenFHE and KeyMemRT are not in nixpkgs, so they come in as
  # flake inputs and are pinned by flake.lock - no manual sha256 to maintain.
  # ?submodules=1 matters for OpenFHE: its CMake runs `git submodule update`,
  # which cannot reach the network from inside the build sandbox, so the
  # submodules (cereal above all) have to arrive with the source.
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    openfhe-fork = {
      url = "git+https://github.com/eymay/openfhe-development?submodules=1";
      flake = false;
    };
    keymemrt = {
      url = "github:eymay/KeyMemRT";
      flake = false;
    };
    # nixpkgs may still carry fmt 11; the project needs 12.
    fmt-src = {
      url = "github:fmtlib/fmt/12.0.0";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, openfhe-fork, keymemrt, fmt-src }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        fmt12 = pkgs.stdenv.mkDerivation {
          pname = "fmt";
          version = "12.0.0";
          src = fmt-src;
          nativeBuildInputs = [ pkgs.cmake ];
          cmakeFlags = [
            "-DFMT_TEST=OFF"
            "-DFMT_DOC=OFF"
            "-DBUILD_SHARED_LIBS=ON"
          ];
          meta.description = "{fmt} 12, the formatting library this project uses";
        };

        # The fork adds a dynamic Q size to evaluation keys, which is what lets
        # KeyMemRT hand hybrid key switching a key truncated to fewer towers.
        # Upstream OpenFHE will not do.
        openfhe = pkgs.stdenv.mkDerivation {
          pname = "openfhe-keymemrt";
          version = "1.2.3-eymay";
          src = openfhe-fork;
          nativeBuildInputs = [ pkgs.cmake pkgs.git ];
          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=Release"
            "-DBUILD_SHARED=ON"
            "-DBUILD_UNITTESTS=OFF"
            "-DBUILD_EXAMPLES=OFF"
            "-DBUILD_BENCHMARKS=OFF"
          ];
          # A full build is long; the parallelism is worth setting explicitly.
          enableParallelBuilding = true;
          meta.description = "OpenFHE, eymay fork, with compressible rotation keys";
        };

        # The frontend itself depends on nothing but {fmt}: it builds a graph,
        # differentiates it and prints MLIR.  No FHE library is involved.
        reboot = pkgs.stdenv.mkDerivation {
          pname = "reboot-keymemrt";
          version = "0.1.0";
          src = ./.;
          nativeBuildInputs = [ pkgs.cmake ];
          buildInputs = [ fmt12 ];
          doCheck = true;
          checkPhase = ''
            runHook preCheck
            ctest --output-on-failure
            runHook postCheck
          '';
          meta.description = "ReBoot training step as ckks-dialect MLIR";
        };
      in {
        packages = {
          default = reboot;
          inherit reboot openfhe fmt12;
        };

        # nix develop
        #
        # Gives the frontend's dependencies plus OpenFHE and the KeyMemRT
        # headers, so `cmake -B build -S . -DKEYMEMRT_DIR=$KEYMEMRT` also builds
        # the key-memory benchmark.
        #
        # keymemrt-opt and keymemrt-translate are NOT built here: they are Bazel
        # targets over a full MLIR/LLVM tree and Bazel fetches from the network,
        # which a Nix build cannot do.  bazelisk is in the shell so you can
        # build them yourself - see the README.
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cmake
            gnumake
            gcc
            clang-tools # clang-format, matching .clang-format
            git
            python3
            bazelisk # for KeyMemRT-Compiler, outside the sandbox
            fmt12
            openfhe
          ];

          shellHook = ''
            export KEYMEMRT="${keymemrt}"
            export OPENFHE_PREFIX="${openfhe}"
            export CMAKE_PREFIX_PATH="${openfhe}:${fmt12}:$CMAKE_PREFIX_PATH"
            echo "reboot-keymemrt dev shell"
            echo "  KEYMEMRT       = $KEYMEMRT"
            echo "  OPENFHE_PREFIX = $OPENFHE_PREFIX"
            echo
            echo "  cmake -B build -S . -DKEYMEMRT_DIR=\$KEYMEMRT"
            echo "  cmake --build build -j && ctest --test-dir build"
          '';
        };

        formatter = pkgs.nixpkgs-fmt;
      });
}
