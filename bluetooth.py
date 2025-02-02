#!/usr/bin/env python3

import dbus
import dbus.exceptions
import dbus.mainloop.glib
import dbus.service

import subprocess
import threading
import array
import os
import socket
import sys

from gi.repository import GLib

# -----------------------------
#   Configuration & Constants
# -----------------------------
BLUEZ_SERVICE_NAME = 'org.bluez'
GATT_MANAGER_IFACE = 'org.bluez.GattManager1'
DBUS_OM_IFACE      = 'org.freedesktop.DBus.ObjectManager'
DBUS_PROP_IFACE    = 'org.freedesktop.DBus.Properties'

GATT_SERVICE_IFACE = 'org.bluez.GattService1'
GATT_CHRC_IFACE    = 'org.bluez.GattCharacteristic1'
GATT_DESC_IFACE    = 'org.bluez.GattDescriptor1'

# The socket to send commands to the “first code”
TRACKER_COMMAND_SOCKET = "/tmp/command_socket"

# This is the socket on which we listen for status messages to notify BLE clients
STATUS_SOCKET = "/tmp/status_socket"

# If the user disconnects from BLE, we want to revert to pre-connected state:
# We'll track “connected_devices” and watch for property changes.

# -----------------------------
#   DBus Exception Classes
# -----------------------------
class InvalidArgsException(dbus.exceptions.DBusException):
    _dbus_error_name = 'org.freedesktop.DBus.Error.InvalidArgs'

class NotSupportedException(dbus.exceptions.DBusException):
    _dbus_error_name = 'org.bluez.Error.NotSupported'

class NotPermittedException(dbus.exceptions.DBusException):
    _dbus_error_name = 'org.bluez.Error.NotPermitted'

class InvalidValueLengthException(dbus.exceptions.DBusException):
    _dbus_error_name = 'org.bluez.Error.InvalidValueLength'

class FailedException(dbus.exceptions.DBusException):
    _dbus_error_name = 'org.bluez.Error.Failed'


# ------------------------------------------
#  Utility: Enable Bluetooth Advertising
# ------------------------------------------
def enable_advertising():
    try:
        subprocess.run(["bluetoothctl", "power", "on"], check=True)
        subprocess.run(["bluetoothctl", "discoverable", "on"], check=True)
        subprocess.run(["sudo", "btmgmt", "advertising", "on"], check=True)
        print("[Bluetooth] Advertising enabled.")
    except subprocess.CalledProcessError as e:
        print(f"[Bluetooth] Failed to enable advertising: {e}")


# ------------------------------------------
#          GATT Application
# ------------------------------------------
class Application(dbus.service.Object):
    """
    org.bluez.GattApplication1 interface implementation
    """
    def __init__(self, bus):
        self.path = '/'
        self.services = []
        dbus.service.Object.__init__(self, bus, self.path)

        # Create our custom ControlService
        self.add_service(ControlService(bus, 0))

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_service(self, service):
        self.services.append(service)

    @dbus.service.method(DBUS_OM_IFACE, out_signature='a{oa{sa{sv}}}')
    def GetManagedObjects(self):
        response = {}
        print('[GattApp] GetManagedObjects')

        for service in self.services:
            response[service.get_path()] = service.get_properties()
            for chrc in service.get_characteristics():
                response[chrc.get_path()] = chrc.get_properties()
                for desc in chrc.get_descriptors():
                    response[desc.get_path()] = desc.get_properties()

        return response


# ------------------------------------------
#              Base Service
# ------------------------------------------
class Service(dbus.service.Object):
    """
    org.bluez.GattService1 interface implementation
    """
    PATH_BASE = '/org/bluez/example/service'

    def __init__(self, bus, index, uuid, primary):
        self.path = self.PATH_BASE + str(index)
        self.bus = bus
        self.uuid = uuid
        self.primary = primary
        self.characteristics = []
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        return {
            GATT_SERVICE_IFACE: {
                'UUID': self.uuid,
                'Primary': self.primary,
                'Characteristics': dbus.Array(
                    self.get_characteristic_paths(),
                    signature='o')
            }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_characteristic(self, chrc):
        self.characteristics.append(chrc)

    def get_characteristics(self):
        return self.characteristics

    def get_characteristic_paths(self):
        return [c.get_path() for c in self.characteristics]

    @dbus.service.method(DBUS_PROP_IFACE,
                         in_signature='s',
                         out_signature='a{sv}')
    def GetAll(self, interface):
        if interface != GATT_SERVICE_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[GATT_SERVICE_IFACE]


class Characteristic(dbus.service.Object):
    """
    org.bluez.GattCharacteristic1 interface implementation
    """
    def __init__(self, bus, index, uuid, flags, service):
        self.path = service.get_path() + '/char' + str(index)
        self.bus = bus
        self.uuid = uuid
        self.flags = flags
        self.service = service
        self.descriptors = []
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        return {
            GATT_CHRC_IFACE: {
                'Service': self.service.get_path(),
                'UUID': self.uuid,
                'Flags': self.flags,
                'Descriptors': dbus.Array(
                    self.get_descriptor_paths(), signature='o')
            }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_descriptor(self, descriptor):
        self.descriptors.append(descriptor)

    def get_descriptors(self):
        return self.descriptors

    def get_descriptor_paths(self):
        return [d.get_path() for d in self.descriptors]

    @dbus.service.method(DBUS_PROP_IFACE,
                         in_signature='s',
                         out_signature='a{sv}')
    def GetAll(self, interface):
        if interface != GATT_CHRC_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[GATT_CHRC_IFACE]

    # Default stubs
    @dbus.service.method(GATT_CHRC_IFACE,
                         in_signature='a{sv}',
                         out_signature='ay')
    def ReadValue(self, options):
        print('[Characteristic] Default ReadValue not supported.')
        raise NotSupportedException()

    @dbus.service.method(GATT_CHRC_IFACE, in_signature='aya{sv}')
    def WriteValue(self, value, options):
        print('[Characteristic] Default WriteValue not supported.')
        raise NotSupportedException()

    @dbus.service.method(GATT_CHRC_IFACE)
    def StartNotify(self):
        print('[Characteristic] Default StartNotify not supported.')
        raise NotSupportedException()

    @dbus.service.method(GATT_CHRC_IFACE)
    def StopNotify(self):
        print('[Characteristic] Default StopNotify not supported.')
        raise NotSupportedException()

    @dbus.service.signal(DBUS_PROP_IFACE,
                         signature='sa{sv}as')
    def PropertiesChanged(self, interface, changed, invalidated):
        pass


class Descriptor(dbus.service.Object):
    """
    org.bluez.GattDescriptor1 interface implementation
    """
    def __init__(self, bus, index, uuid, flags, characteristic):
        self.path = characteristic.get_path() + '/desc' + str(index)
        self.bus = bus
        self.uuid = uuid
        self.flags = flags
        self.chrc = characteristic
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        return {
            GATT_DESC_IFACE: {
                'Characteristic': self.chrc.get_path(),
                'UUID': self.uuid,
                'Flags': self.flags,
            }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    @dbus.service.method(DBUS_PROP_IFACE,
                         in_signature='s',
                         out_signature='a{sv}')
    def GetAll(self, interface):
        if interface != GATT_DESC_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[GATT_DESC_IFACE]

    @dbus.service.method(GATT_DESC_IFACE,
                         in_signature='a{sv}',
                         out_signature='ay')
    def ReadValue(self, options):
        print('[Descriptor] Default ReadValue not supported.')
        raise NotSupportedException()

    @dbus.service.method(GATT_DESC_IFACE, in_signature='aya{sv}')
    def WriteValue(self, value, options):
        print('[Descriptor] Default WriteValue not supported.')
        raise NotSupportedException()


# ------------------------------------------
#   Helper to send commands to the tracker
# ------------------------------------------
def send_command_to_tracker(command: str):
    """
    Sends a command (e.g., 'STOP', 'CALIBRATE') to the first code 
    via the Unix domain socket at /tmp/command_socket.
    """
    if not command:
        return
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(TRACKER_COMMAND_SOCKET)
            s.sendall(command.encode('utf-8'))
    except ConnectionRefusedError:
        print(f"[Bluetooth] Could not connect to tracker socket. Is the tracker running?")
    except FileNotFoundError:
        print(f"[Bluetooth] Socket {TRACKER_COMMAND_SOCKET} not found.")
    except Exception as e:
        print(f"[Bluetooth] Error sending command to tracker: {e}")


# ------------------------------------------------
#   Custom Service & Characteristics
# ------------------------------------------------
class ControlService(Service):
    """
    GATT Service for controlling the distance tracker (start, stop, calibrate).
    """
    CONTROL_SVC_UUID = '12345678-1234-5678-1234-56789abcdef0'  # Example

    def __init__(self, bus, index):
        Service.__init__(self, bus, index, self.CONTROL_SVC_UUID, True)
        # Add a write characteristic that receives commands
        self.control_chrc = ControlCharacteristic(bus, 0, self)
        self.add_characteristic(self.control_chrc)

        # Add a notify characteristic to send status updates
        self.status_chrc = StatusCharacteristic(bus, 1, self)
        self.add_characteristic(self.status_chrc)

    # Helper so a characteristic can find the status char
    def get_characteristic_by_uuid(self, uuid):
        for ch in self.characteristics:
            if ch.uuid == uuid:
                return ch
        return None


class ControlCharacteristic(Characteristic):
    CONTROL_CHRC_UUID = '12345678-1234-5678-1234-56789abcdef1'

    def __init__(self, bus, index, service):
        super().__init__(bus, index, self.CONTROL_CHRC_UUID,
                         ['write'], service)
        self.process = None

    def WriteValue(self, value, options):
        """
        Receive a command: 'start:<distance>', 'stop', 'calibrate', etc.
        """
        try:
            command_str = ''.join([chr(byte) for byte in value]).strip().lower()
            parts = command_str.split(':')
        except UnicodeDecodeError:
            print('[ControlCharacteristic] Non-UTF8 data received.')
            raise InvalidArgsException('Invalid encoding.')

        if parts[0] == 'start':
            distance = parts[1] if len(parts) > 1 else '0.5'
            self.handle_start(distance)
        elif parts[0] == 'stop':
            self.handle_stop()
        elif parts[0] == 'calibrate':
            self.handle_calibrate()
        else:
            print(f"[ControlCharacteristic] Unknown command: {command_str}")
            raise InvalidArgsException('Invalid command')

    def handle_start(self, distance):
        print(f"[ControlCharacteristic] START with distance={distance}")
        # Option 1: Launch the first code as a separate process
        #           (the original approach).
        # Or use a direct socket approach if first code is already running.
        # We'll keep your original approach:
        if self.process and self.process.poll() is None:
            print("[ControlCharacteristic] Another process is running. Stop it first or skip.")
            return

        self.process = subprocess.Popen(
            ['python3', 'control3.py', str(distance)],
            stderr=subprocess.PIPE,
            text=True
        )
        # Monitor the process in a thread
        threading.Thread(target=self.monitor_process, daemon=True).start()

    def handle_stop(self):
        print("[ControlCharacteristic] STOP command received.")
        # Instead of sending 'q' via pyautogui, we send a direct command
        send_command_to_tracker("STOP")
        # If we want to forcibly kill the process, do it here:
        # if self.process and self.process.poll() is None:
        #     self.process.terminate()
        #     print("[ControlCharacteristic] Killed the poseDetection1.py process.")

    def handle_calibrate(self):
        print("[ControlCharacteristic] CALIBRATE command received.")
        # Instead of sending 'c' via pyautogui, send direct command
        send_command_to_tracker("CALIBRATE")

    def monitor_process(self):
        """
        If poseDetection1.py fails or exits, we can notify the StatusCharacteristic.
        """
        if not self.process:
            return
        stdout, stderr = self.process.communicate()
        rc = self.process.returncode
        status_chrc = self.service.get_characteristic_by_uuid(StatusCharacteristic.STATUS_CHRC_UUID)
        if status_chrc:
            if rc != 0:
                error_message = f"TrackingFailed: {stderr.strip() or 'Unknown error'}"
                status_chrc.update_status(error_message)
            else:
                status_chrc.update_status("TrackingEnded")


class StatusCharacteristic(Characteristic):
    """
    Notifies BLE clients about status changes. 
    We'll listen on /tmp/status_socket for updates from the first code.
    """
    STATUS_CHRC_UUID = '12345678-1234-5678-1234-56789abcdef2'

    def __init__(self, bus, index, service):
        super().__init__(bus, index, self.STATUS_CHRC_UUID,
                         ['notify'], service)
        self.notifying = False
        self.listen_thread = None

    def StartNotify(self):
        print("[StatusCharacteristic] StartNotify called.")
        if self.notifying:
            print("[StatusCharacteristic] Already notifying.")
            return
        self.notifying = True
        self.listen_thread = threading.Thread(target=self.listen_for_status, daemon=True)
        self.listen_thread.start()
        print("[StatusCharacteristic] Started status notifications.")

    def StopNotify(self):
        print("[StatusCharacteristic] StopNotify called.")
        if not self.notifying:
            return
        self.notifying = False
        print("[StatusCharacteristic] Stopped status notifications.")

    def listen_for_status(self):
        """
        Listens on /tmp/status_socket for status from the first code.
        If the code sends messages, we notify the BLE client.
        """
        if os.path.exists(STATUS_SOCKET):
            os.remove(STATUS_SOCKET)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as srv:
            srv.bind(STATUS_SOCKET)
            srv.listen(1)
            print(f"[StatusCharacteristic] Listening on {STATUS_SOCKET} for status messages.")
            while self.notifying:
                try:
                    conn, _ = srv.accept()
                    with conn:
                        data = conn.recv(1024).decode()
                        if data:
                            print(f"[StatusCharacteristic] Received status: {data}")
                            self.update_status(data)
                except Exception as ex:
                    print(f"[StatusCharacteristic] Error: {ex}")
                    if self.notifying:
                        time.sleep(1)

    def listen_for_status(self):
        if os.path.exists(STATUS_SOCKET):
            os.remove(STATUS_SOCKET)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server_sock:
            server_sock.bind(STATUS_SOCKET)
            server_sock.listen(1)
            while self.notifying:
                conn, _ = server_sock.accept()
                with conn:
                    status = conn.recv(1024).decode()
                    if status:
                        self.update_status(status)

    def update_status(self, status):
        if not self.notifying:
            print('Notifications not enabled, cannot send status update')
            return

        # Convert status string to list of dbus.Byte
        try:
            value = [dbus.Byte(c.encode()) for c in status]
        except AttributeError:
            print('Status must be a string')
            return

        print(f'Updating status: {status}')
        self.PropertiesChanged(GATT_CHRC_IFACE, { 'Value': value }, [])

# ----------------------------------------------
#  Detecting BLE Device Disconnections
# ----------------------------------------------
connected_devices = set()

def adapter_properties_changed(interface, changed, invalidated, path=None):
    """
    Example: We can watch 'org.bluez.Device1' for 'Connected' changes
    to track device connections/disconnections.
    """
    if interface != "org.bluez.Device1":
        return

    if 'Connected' in changed:
        is_connected = changed['Connected']
        if is_connected:
            connected_devices.add(path)
            print(f"[DBus] Device {path} connected. Current set: {connected_devices}")
        else:
            if path in connected_devices:
                connected_devices.remove(path)
            print(f"[DBus] Device {path} disconnected. Reverting to pre-connected state.")
            # If you want to do something on disconnect, e.g. reset app state:
            # For example, kill the external poseDetection1.py or send STOP:
            send_command_to_tracker("STOP")
            # Possibly unexport the application or do more logic, if you wish.


# ----------------------------------------------
#        Main DBus / GATT Registration
# ----------------------------------------------
def register_app_cb():
    print('[Main] GATT application registered')

def register_app_error_cb(error):
    print('[Main] Failed to register application:', error)
    mainloop.quit()

def find_adapter(bus):
    remote_om = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, '/'),
                               DBUS_OM_IFACE)
    objects = remote_om.GetManagedObjects()

    for o, props in objects.items():
        if GATT_MANAGER_IFACE in props.keys():
            return o
    return None

def main():
    global mainloop

    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()

    # Listen for device connect/disconnect
    bus.add_signal_receiver(
        adapter_properties_changed,
        dbus_interface="org.freedesktop.DBus.Properties",
        signal_name="PropertiesChanged",
        path_keyword="path"
    )

    adapter = find_adapter(bus)
    if not adapter:
        print('[Main] GattManager1 interface not found.')
        return

    # Create GATT app
    app = Application(bus)
    service_manager = dbus.Interface(
        bus.get_object(BLUEZ_SERVICE_NAME, adapter),
        GATT_MANAGER_IFACE
    )

    # Enable advertising
    enable_advertising()

    # Register GATT application
    print('[Main] Registering GATT application...')
    service_manager.RegisterApplication(
        app.get_path(),
        {},
        reply_handler=register_app_cb,
        error_handler=register_app_error_cb
    )

    mainloop = GLib.MainLoop()
    mainloop.run()

if __name__ == '__main__':
    main()
