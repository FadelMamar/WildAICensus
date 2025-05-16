@REM Run in administrator mode

start /w "" "C:\Users\FADELCO\Downloads\Docker Desktop Installer.exe" install -accept-license ^
   --installation-dir="D:\Docker\Docker" ^
   --wsl-default-data-root="D:\Docker\wsl" ^
   --windows-containers-default-data-root="D:\\Docker"
