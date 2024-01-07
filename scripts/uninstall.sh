#!/bin/sh
# This script uninstalls Ollama.

set -eu

# default error condition
error() { echo "ERROR $*"; exit 1; }

# Validate a command exists
available() { command -v "$1" >/dev/null; }

# Validate script is running on Linux
[ "$(uname -s)" = "Linux" ] || error 'This script is intended to run on Linux only.'

# Make sure script is run as root
# if user is running as root $SUDO will be set to empty string
SUDO=
if [ "$(id -u)" -ne 0 ]; then
    # Make sure sudo is installed on the system
    # if not, ask the user to run as root and exit
    if ! available sudo; then
        error "Please re-run this script as root"
    fi

    SUDO="sudo"
fi

# Confirmation prompt
echo "Are you sure you want to uninstall Ollama? (y/N): "
read -r REPLY
case $REPLY in
    [Yy]* ) 
        ;;
    * ) 
        echo "Uninstallation cancelled."
        exit 1
        ;;
esac

# Remove stop and remove any systemd services if they exist
if [ -f '/etc/systemd/system/ollama.service' ]; then

    # inform user we are stopping and disabling the service at startup
    echo "Stopping and disabling ollama service at system startup..."

    # Stop and disable the ollama service if its running
    systemctl is-enabled --quiet ollama && $SUDO systemctl disable --now --quiet ollama 

    # Delete the ollama unit file
    echo "Deleting the ollama service file..."
    $SUDO rm -f /etc/systemd/system/ollama.service

    # Validate system is up to date
    $SUDO systemctl daemon-reload

    # Inform user ollama is stopped and disabled
    echo "Ollama service stopped and removed."
fi

# Delete the ollama binary if it exists
if available ollama; then
    echo "Deleting the ollama binary..."
    $SUDO rm -f "$(command -v ollama)"
fi

# Remove any locally installed models if they exist
if [ -d '/usr/share/ollama' ]; then
    echo "Deleting locally installed Ollama models..."
    $SUDO rm -rf /usr/share/ollama/
fi

# Remove the ollama group and user from the system if they exist
if getent passwd ollama > /dev/null; then
    echo "Deleting the ollama user..."
    $SUDO userdel ollama > /dev/null 2>&1
fi

if getent group ollama > /dev/null; then
    echo "Deleting the ollama group..."
    $SUDO groupdel ollama > /dev/null 2>&1
fi

# Inform user ollama is uninstalled
echo "Ollama has been successfully uninstalled from the system."
