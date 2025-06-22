# Connecting a Bluetooth Speaker on Raspberry Pi (CLI Guide)

This guide provides a complete command-line workflow for pairing a Bluetooth speaker with your Raspberry Pi, testing the connection, and basic troubleshooting.

## Step 1: Preparation

Before starting, you must make your Bluetooth speaker discoverable.

1.  **Turn on your Bluetooth speaker.**
2.  **Put it into Pairing Mode.** This process varies by device but usually involves pressing and holding a Bluetooth button until a light starts flashing or a sound is made.

## Step 2: Find Your Speaker's MAC Address

We will use the `bluetoothctl` utility to find and manage Bluetooth devices.

1.  Open a terminal on your Raspberry Pi.
2.  Launch the Bluetooth control tool by typing:
    ```bash
    bluetoothctl
    ```
    Your command prompt will change to `[bluetooth]#`.
3.  Start scanning for nearby devices.
    ```bash
    scan on
    ```
4.  Watch the output. You will see a list of devices appearing. Look for your speaker's name. The line will look something like this:
    ```
    [NEW] Device XX:XX:XX:XX:XX:XX My Speaker Name
    ```
5.  Copy the alphanumeric code (`XX:XX:XX:XX:XX:XX`). This is the **MAC address** of your speaker.
6.  Once you have the address, stop scanning to prevent clutter:
    ```bash
    scan off
    ```

## Step 3: Pair, Connect, and Trust the Speaker

Now, while still inside the `bluetoothctl` tool, use the MAC address you just copied to establish the connection.

1.  **Pair with the device.** This creates a security key to allow the connection.
    ```bash
    # Replace XX:XX:XX:XX:XX:XX with your speaker's MAC address
    pair XX:XX:XX:XX:XX:XX
    ```
2.  **Connect to the device.** This establishes the active link.
    ```bash
    connect XX:XX:XX:XX:XX:XX
    ```
3.  **Trust the device.** This is an important step that tells your Raspberry Pi to automatically reconnect to this speaker in the future.
    ```bash
    trust XX:XX:XX:XX:XX:XX
    ```
4.  You can now safely leave the utility.
    ```bash
    exit
    ```

## Step 4: Test the Audio Output

With the speaker connected, you can test if the audio is working correctly from the terminal.

### Option A: The `speaker-test` Utility

This command generates a "pink noise" sound to test speakers.

```bash
# -c 2 tests stereo (left and right channels)
# -t pink specifies the type of sound
# -l 1 runs the test once
speaker-test -c 2 -t pink -l 1
```

You should hear a brief hissing/static sound from your Bluetooth speaker.
