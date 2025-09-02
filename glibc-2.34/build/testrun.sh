#!/bin/bash
builddir=`dirname "$0"`
GCONV_PATH="${builddir}/iconvdata"

usage () {
cat << EOF
Usage: $0 [OPTIONS] <program> [ARGUMENTS...]

  --tool=TOOL  Run with the specified TOOL. It can be strace, valgrind or
               container. The container will run within support/test-container.
EOF

  exit 1
}

toolname=default
while test $# -gt 0 ; do
  case "$1" in
    --tool=*)
      toolname="${1:7}"
      shift
      ;;
    --*)
      usage
      ;;
    *)
      break
      ;;
  esac
done

if test $# -eq 0 ; then
  usage
fi

case "$toolname" in
  default)
    exec   env GCONV_PATH="${builddir}"/iconvdata LOCPATH="${builddir}"/localedata LC_ALL=C  "${builddir}"/elf/ld-linux-x86-64.so.2 --library-path "${builddir}":"${builddir}"/math:"${builddir}"/elf:"${builddir}"/dlfcn:"${builddir}"/nss:"${builddir}"/nis:"${builddir}"/rt:"${builddir}"/resolv:"${builddir}"/mathvec:"${builddir}"/support:"${builddir}"/crypt:"${builddir}"/nptl ${1+"$@"}
    ;;
  strace)
    exec strace  -EGCONV_PATH=/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/iconvdata  -ELOCPATH=/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/localedata  -ELC_ALL=C  /teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/elf/ld-linux-x86-64.so.2 --library-path /teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/math:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/elf:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/dlfcn:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/nss:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/nis:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/rt:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/resolv:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/mathvec:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/support:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/crypt:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/nptl ${1+"$@"}
    ;;
  valgrind)
    exec env GCONV_PATH=/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/iconvdata LOCPATH=/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/localedata LC_ALL=C valgrind  /teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/elf/ld-linux-x86-64.so.2 --library-path /teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/math:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/elf:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/dlfcn:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/nss:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/nis:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/rt:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/resolv:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/mathvec:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/support:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/crypt:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/nptl ${1+"$@"}
    ;;
  container)
    exec env GCONV_PATH=/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/iconvdata LOCPATH=/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/localedata LC_ALL=C  /teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/elf/ld-linux-x86-64.so.2 --library-path /teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/math:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/elf:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/dlfcn:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/nss:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/nis:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/rt:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/resolv:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/mathvec:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/support:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/crypt:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/nptl /teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/support/test-container env GCONV_PATH=/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/iconvdata LOCPATH=/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/localedata LC_ALL=C  /teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/elf/ld-linux-x86-64.so.2 --library-path /teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/math:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/elf:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/dlfcn:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/nss:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/nis:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/rt:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/resolv:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/mathvec:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/support:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/crypt:/teamspace/studios/this_studio/AI-Automated-Radiology-Report/glibc-2.34/build/nptl ${1+"$@"}
    ;;
  *)
    usage
    ;;
esac
