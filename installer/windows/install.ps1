# OpenAlgo Atlas setup for Windows.
#
# Run by OpenAlgoAtlasSetup.exe (or directly: right-click, Run with PowerShell). Safe to run again: it is also how
# OpenAlgo Atlas is updated. It never touches your settings (.env), your account and broker keys (db\) or the
# Atlas market data (atlas_engine\data); it only replaces the program files.
#
# What it does:
#   1. downloads uv, the tool that installs Python and OpenAlgo's libraries, into <folder>\tools
#   2. downloads the latest OpenAlgo Atlas from GitHub (irtteja/openalgo, branch atlas)
#   3. installs Python and the libraries (uv sync)
#   4. makes a desktop icon, and optionally starts OpenAlgo when you sign in to Windows

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # the download progress bar makes PowerShell 5 downloads very slow
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Repo    = 'irtteja/openalgo'
$Branch  = 'atlas'
$Dir     = Join-Path $env:USERPROFILE 'OpenAlgoAtlas'
$Tools   = Join-Path $Dir 'tools'
$Uv      = Join-Path $Tools 'uv.exe'
$Work    = Join-Path $env:TEMP ('openalgo-atlas-setup-' + [guid]::NewGuid().ToString('N'))
$Launch  = Join-Path $Dir 'installer\windows\start_openalgo.cmd'

function Say($msg)  { Write-Host '' ; Write-Host $msg -ForegroundColor Cyan }
function Done($msg) { Write-Host "  $msg" -ForegroundColor Green }
function Stop-Setup($msg) {
    Write-Host ''
    Write-Host $msg -ForegroundColor Yellow
    Write-Host ''
    Read-Host 'Press Enter to close this window'
    exit 1
}

function Test-OpenAlgoRunning {
    try {
        $c = New-Object Net.Sockets.TcpClient
        $c.Connect('127.0.0.1', 5000)
        $c.Close()
        return $true
    } catch { return $false }
}

function New-Shortcut($path, $target, $workdir, $desc, $minimized) {
    $sh = New-Object -ComObject WScript.Shell
    $s = $sh.CreateShortcut($path)
    $s.TargetPath = $target
    $s.WorkingDirectory = $workdir
    $s.Description = $desc
    $s.IconLocation = "$env:SystemRoot\System32\SHELL32.dll,13"
    if ($minimized) { $s.WindowStyle = 7 }
    $s.Save()
}

Write-Host '============================================================'
Write-Host '  OpenAlgo Atlas setup'
Write-Host "  Installs into: $Dir"
Write-Host '  This takes about 5 to 10 minutes. Keep this window open.'
Write-Host '============================================================'

if (Test-OpenAlgoRunning) {
    Stop-Setup ('OpenAlgo is running right now. Close its window (the black window titled OpenAlgo Atlas), ' +
                'then run this setup again.')
}

try {
    New-Item -ItemType Directory -Force -Path $Dir, $Tools, $Work | Out-Null

    Say 'Step 1 of 4: downloading uv (installs Python for OpenAlgo) ...'
    $uvZip = Join-Path $Work 'uv.zip'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile $uvZip
    Expand-Archive -Path $uvZip -DestinationPath (Join-Path $Work 'uv') -Force
    Get-ChildItem -Path (Join-Path $Work 'uv') -Recurse -Include 'uv.exe', 'uvx.exe' | ForEach-Object {
        Copy-Item $_.FullName -Destination $Tools -Force
    }
    if (-not (Test-Path $Uv)) { throw 'uv.exe was not found in the download.' }
    Done 'done'

    Say 'Step 2 of 4: downloading the latest OpenAlgo Atlas ...'
    $srcZip = Join-Path $Work 'openalgo.zip'
    Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/$Repo/archive/refs/heads/$Branch.zip" -OutFile $srcZip
    Expand-Archive -Path $srcZip -DestinationPath (Join-Path $Work 'src') -Force
    $top = Get-ChildItem -Path (Join-Path $Work 'src') -Directory | Select-Object -First 1
    # Copy over the old program files. Nothing is deleted, and the package holds no .env, db\ or atlas data,
    # so your settings, account and downloaded market data stay as they are.
    $rc = Start-Process -FilePath robocopy.exe -ArgumentList @("`"$($top.FullName)`"", "`"$Dir`"", '/E', '/NFL', '/NDL', '/NJH', '/NJS', '/NP', '/R:2', '/W:2') -NoNewWindow -Wait -PassThru
    if ($rc.ExitCode -ge 8) { throw "Copying the program files failed (robocopy code $($rc.ExitCode))." }
    Done 'done'

    Say 'Step 3 of 4: installing Python and the libraries OpenAlgo needs (the longest step) ...'
    Push-Location $Dir
    & $Uv sync
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { throw 'Installing the libraries did not finish.' }
    Done 'done'

    Say 'Step 4 of 4: making the desktop icon ...'
    $desktop = [Environment]::GetFolderPath('Desktop')
    New-Shortcut (Join-Path $desktop 'OpenAlgo Atlas.lnk') $Launch $Dir 'Start OpenAlgo Atlas' $false
    $upd = Join-Path $Dir 'installer\windows\update_openalgo.cmd'
    New-Shortcut (Join-Path $desktop 'Update OpenAlgo Atlas.lnk') $upd $Dir 'Download the latest OpenAlgo Atlas' $false
    Done 'desktop icons: OpenAlgo Atlas, Update OpenAlgo Atlas'

    $startup = Join-Path ([Environment]::GetFolderPath('Startup')) 'OpenAlgo Atlas.lnk'
    Write-Host ''
    Write-Host 'The Atlas updates itself every weekday at 7:30 pm, but only while OpenAlgo is running.'
    $ans = Read-Host 'Start OpenAlgo automatically when you sign in to Windows? Type Y or N, then Enter (recommended: Y)'
    if ($ans -match '^[Yy]') {
        New-Shortcut $startup $Launch $Dir 'Start OpenAlgo Atlas' $true
        Done 'OpenAlgo will start (minimised) each time you sign in to Windows'
    } elseif (Test-Path $startup) {
        Remove-Item $startup -Force
        Done 'automatic start is off'
    }
} catch {
    Stop-Setup ("Setup could not finish: $($_.Exception.Message)`n" +
                'Check the internet connection and run the setup again; it continues from where it stopped.')
} finally {
    Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
}

Write-Host ''
Write-Host '============================================================' -ForegroundColor Green
Write-Host '  OpenAlgo Atlas is installed.' -ForegroundColor Green
Write-Host '  Starting it now; your browser opens at http://127.0.0.1:5000' -ForegroundColor Green
Write-Host '  From now on, use the OpenAlgo Atlas icon on your desktop.' -ForegroundColor Green
Write-Host '============================================================' -ForegroundColor Green
Start-Process -FilePath $Launch -WorkingDirectory $Dir
Start-Sleep -Seconds 5
