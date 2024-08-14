Having a running environment for an Intel CPU has proven quite complex.

The issue is related to torch, which standard installation does not run for unknown reasons
(there is an unresolved DLL, but according to the comments online, this is not the issue).

In the end, it has been necessary to:
- install miniconda
- install oneapi deep network library from Intel site

https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=windows&windows-install-type=online

- install python 3.11

    conda install python=3.11

Note: don't install Intel torch via conda, because the packet is for Linux only!

Also remember to enable the long path support in Windows:
    . run regedit
    . navigate to HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
    . set LongPathsEnabled to 1, create the key if it doesn’t exist
