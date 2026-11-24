; Builds OpenAlgoAtlasSetup.exe: a small wrapper that runs install.ps1, which downloads and installs everything.
; Build on Linux or Windows:  makensis installer/windows/OpenAlgoAtlasSetup.nsi
Unicode true
Name "OpenAlgo Atlas"
Caption "OpenAlgo Atlas setup"
OutFile "OpenAlgoAtlasSetup.exe"
RequestExecutionLevel user
SilentInstall silent
VIProductVersion "1.0.0.0"
VIAddVersionKey "ProductName" "OpenAlgo Atlas setup"
VIAddVersionKey "FileDescription" "Installs OpenAlgo with the Atlas market analysis"
VIAddVersionKey "FileVersion" "1.0.0"

Section
  InitPluginsDir
  File "/oname=$PLUGINSDIR\install.ps1" "install.ps1"
  ExecWait 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$PLUGINSDIR\install.ps1"'
SectionEnd
