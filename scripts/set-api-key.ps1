# Asks for your Anthropic API key (typing is hidden) and saves it for your Windows user.
# Run:  powershell -ExecutionPolicy Bypass -File scripts\set-api-key.ps1
#
# It is saved twice: as "pilensekey" (your name for it) and as "ANTHROPIC_API_KEY"
# (the name the Anthropic library, and so `pilens evolve`, actually reads).
# Nothing is written into the project folder.

$secure = Read-Host "Paste your Anthropic API key (input is hidden)" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr).Trim()
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}

if (-not $key) {
    Write-Host "No key entered. Nothing saved." -ForegroundColor Yellow
    exit 1
}
if (-not $key.StartsWith("sk-ant-")) {
    Write-Host "That doesn't look like an Anthropic key (they start with 'sk-ant-'). Nothing saved." -ForegroundColor Yellow
    exit 1
}

foreach ($name in @("pilensekey", "ANTHROPIC_API_KEY")) {
    [Environment]::SetEnvironmentVariable($name, $key, "User")  # permanent, for new terminals
    Set-Item -Path "Env:$name" -Value $key                       # this terminal too
}

$shown = $key.Substring(0, 10) + "..." + $key.Substring($key.Length - 4)
Write-Host "Saved $shown as pilensekey and ANTHROPIC_API_KEY for your user." -ForegroundColor Green
Write-Host "Open a new terminal (or restart the Claude app) before running: pilens evolve --rounds 3"
