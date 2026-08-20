# Fallback for setups without flakes enabled.
#
#   nix-shell
#
# It evaluates flake.nix through flake-compat, so the environment has one
# definition rather than two that drift apart.  With flakes enabled prefer
#
#   nix develop
#
{ system ? builtins.currentSystem }:

let
  flake-compat = builtins.fetchTarball {
    url = "https://github.com/edolstra/flake-compat/archive/master.tar.gz";
  };
in
(import flake-compat { src = ./.; inherit system; }).shellNix
