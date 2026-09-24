#!/usr/bin/bash

# =============================================================================
# nbia_data_retriever_install.sh
#
# Description:
# Installs the National Biomedical Imaging Archive (NBIA) Data Retriever. The
# NBIA Data Retriever is used to download data from The Cancer Imaging Archive
# (TCIA). This script is intended for use inside the devcontainer that
# accompanies this repository.
#
# This process is mostly derived from the instructions here:
# https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Command-Line+Interface+Guide
#
# Usage:
#
# bash nbia_data_retriever_install.sh /full/path/to/download_location
#
# =============================================================================

# Check if the first argument is empty
if [ -z "$1" ]; then
  echo "First argument required."
  echo ""
  echo "Usage:"
  echo "bash nbia_data_retriever_install.sh /full/path/to/download_location"
  exit 1
fi

echo "Downloading NBIA Data Retriever..."

# Note: newer versions of the NBIA data retriever are started to be hosted on
# GitHub here: https://github.com/CBIIT/NBIA-TCIA/releases, however as of
# 10/20/2024, the linux .deb and .rpm packages are still in pre-release.
#
# TODO: Change this to the latest valid Debian release once NBIA releases from
# GitHub are mature enough.
installer_web_location="https://cbiit-download.nci.nih.gov/nbia/releases/ForTCIA/NBIADataRetriever_4.4/nbia-data-retriever-4.4.2.deb"
installer_downloads="$1"
installer_local_location="$installer_downloads/nbia-data-retriever-4.4.2.deb"

if [ -e "$installer_local_location" ]; then
    echo "NBIA installer exists. Proceeding with install."
else
    echo "NBIA installer not found. Attempting to download."
    wget -P "$installer_downloads" "$installer_web_location"

    if [ $? -eq 0 ]; then
        echo "Downloaded NBIA Data Retriever!"
    else
        echo "Failed to download NBIA Data Retriever."
        exit 1
    fi
fi


# The NBIA Data Retriever installer requires a desktop directory to run. If
# this directory is not present the installer will error when attempting to
# make a shortcut in this directory. The shortcut itself is not needed since
# this repository uses the CLI for the NBIA Data Retriever.

echo "Creating desktop directory needed by install and running installer..."

sudo mkdir -p /usr/share/desktop-directories/ && \
sudo -S dpkg -i "$installer_local_location"

# Check the install location by checking for the executable since due to a
# known bug in the installer's dependencies, it will error out when attempting
# to install in headless Ubuntu, even though the installer finished
# successfully.
nbia_install_location="/opt/nbia-data-retriever/nbia-data-retriever"

if [ -e "$nbia_install_location" ]; then
    echo "Installed NBIA Data Retriever!"
else
    echo "Failed to install NBIA Data Retriever."
    exit 1
fi

exit 0
