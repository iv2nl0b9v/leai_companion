git clone https://github.com/respeaker/seeed-voicecard
cd seeed-voicecard
echo "The driver is not compatible with the new linux kernel, so the script will suggest to revert it."
sudo ./install_arm64.sh
echo "The kernel has been updated, so you need to run install.sh again after this reboot."
sudo reboot now
