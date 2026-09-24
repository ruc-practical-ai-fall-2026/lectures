#!/usr/bin/bash

# =============================================================================
# clear_downloads.sh
#
# Description:
# Convenience script to clear all downloads in the given folder.
#
# Usage:
#
# bash clear_downloads.sh
#
# =============================================================================

SCRIPTS_FOLDER="/workspaces/data-loaders/scripts"
NBIA_DOWNLOADS_FOLDER="/workspaces/data-loaders/downloads/nbia"

confirm_action() {
    read -p "Are you sure you want to proceed? [Y/n]: " response
    case $response in
        [Y]* ) return 0;;
        [Nn]* ) return 1;;
        * )
            echo "Please answer 'Y' (yes) or 'n' (no)."
            confirm_action
            ;;
    esac
}

echo "This will remove all files in $NBIA_DOWNLOADS_FOLDER."

if confirm_action; then
    echo "Clearing NBIA Downloads folder in $NBIA_DOWNLOADS_FOLDER..."
    rm -rf "$NBIA_DOWNLOADS_FOLDER"/*
    echo "Removed NBIA downloads."
else
    echo "Quitting."
fi

exit 0