import dbus
import dbus.exceptions
import dbus.mainloop.glib
import dbus.service
import subprocess
import array
import pyautogui
from gi.repository import GLib  # Updated import
import sys

from random import randint

mainloop = None
def enable_advertising():
    try:
        subprocess.run(["bluetoothctl", "power", "on"], check=True)
        subprocess.run(["bluetoothctl", "discoverable", "on"], check=True)
        # Run the command with sudo
        subprocess.run(['sudo', 'btmgmt', 'advertising', 'on'], check=True)
        print('Advertising enabled successfully.')
    except subprocess.CalledProcessError as e:
        print(f'Failed to enable advertising: {e}')

BLUEZ_SERVICE_NAME = 'org.bluez'
GATT_MANAGER_IFACE = 'org.bluez.GattManager1'
DBUS_OM_IFACE =      'org.freedesktop.DBus.ObjectManager'
DBUS_PROP_IFACE =    'org.freedesktop.DBus.Properties'

GATT_SERVICE_IFACE = 'org.bluez.GattService1'
GATT_CHRC_IFACE =    'org.bluez.GattCharacteristic1'
GATT_DESC_IFACE =    'org.bluez.GattDescriptor1'

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


class Application(dbus.service.Object):
    """
    org.bluez.GattApplication1 interface implementation
    """
    def __init__(self, bus):
        self.path = '/'
        self.services = []
        dbus.service.Object.__init__(self, bus, self.path)
        self.add_service(ControlService(bus, 0))

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_service(self, service):
        self.services.append(service)

    @dbus.service.method(DBUS_OM_IFACE, out_signature='a{oa{sa{sv}}}')
    def GetManagedObjects(self):
        response = {}
        print('GetManagedObjects')

        for service in self.services:
            response[service.get_path()] = service.get_properties()
            chrcs = service.get_characteristics()
            for chrc in chrcs:
                response[chrc.get_path()] = chrc.get_properties()
                descs = chrc.get_descriptors()
                for desc in descs:
                    response[desc.get_path()] = desc.get_properties()

        return response


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

    def add_characteristic(self, characteristic):
        self.characteristics.append(characteristic)

    def get_characteristic_paths(self):
        result = []
        for chrc in self.characteristics:
            result.append(chrc.get_path())
        return result

    def get_characteristics(self):
        return self.characteristics

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
        self.path = service.path + '/char' + str(index)
        self.bus = bus
        self.uuid = uuid
        self.service = service
        self.flags = flags
        self.descriptors = []
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        return {
                GATT_CHRC_IFACE: {
                        'Service': self.service.get_path(),
                        'UUID': self.uuid,
                        'Flags': self.flags,
                        'Descriptors': dbus.Array(
                                self.get_descriptor_paths(),
                                signature='o')
                }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_descriptor(self, descriptor):
        self.descriptors.append(descriptor)

    def get_descriptor_paths(self):
        result = []
        for desc in self.descriptors:
            result.append(desc.get_path())
        return result

    def get_descriptors(self):
        return self.descriptors

    @dbus.service.method(DBUS_PROP_IFACE,
                         in_signature='s',
                         out_signature='a{sv}')
    def GetAll(self, interface):
        if interface != GATT_CHRC_IFACE:
            raise InvalidArgsException()

        return self.get_properties()[GATT_CHRC_IFACE]

    @dbus.service.method(GATT_CHRC_IFACE,
                        in_signature='a{sv}',
                        out_signature='ay')
    def ReadValue(self, options):
        print('Default ReadValue called, returning error')
        raise NotSupportedException()

    @dbus.service.method(GATT_CHRC_IFACE, in_signature='aya{sv}')
    def WriteValue(self, value, options):
        print('Default WriteValue called, returning error')
        raise NotSupportedException()

    @dbus.service.method(GATT_CHRC_IFACE)
    def StartNotify(self):
        print('Default StartNotify called, returning error')
        raise NotSupportedException()

    @dbus.service.method(GATT_CHRC_IFACE)
    def StopNotify(self):
        print('Default StopNotify called, returning error')
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
        self.path = characteristic.path + '/desc' + str(index)
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
        print ('Default ReadValue called, returning error')
        raise NotSupportedException()

    @dbus.service.method(GATT_DESC_IFACE, in_signature='aya{sv}')
    def WriteValue(self, value, options):
        print('Default WriteValue called, returning error')
        raise NotSupportedException()


class ControlService(Service):
    """
    Control Service that allows sending start, stop, and calibrate commands.
    """
    CONTROL_SVC_UUID = '12345678-1234-5678-1234-56789abcdef0'  # Replace with your UUID

    def __init__(self, bus, index):
        Service.__init__(self, bus, index, self.CONTROL_SVC_UUID, True)
        self.add_characteristic(ControlCharacteristic(bus, 0, self))
        # Optionally, add a status characteristic to notify clients about status updates
        self.add_characteristic(StatusCharacteristic(bus, 1, self))


class ControlCharacteristic(Characteristic):
    """
    Writable characteristic to receive start, stop, and calibrate commands.
    """
    CONTROL_CHRC_UUID = '12345678-1234-5678-1234-56789abcdef1'  # Replace with your UUID

    def __init__(self, bus, index, service):
        Characteristic.__init__(
                self, bus, index,
                self.CONTROL_CHRC_UUID,
                ['write'],
                service)
        self.value = []

    def WriteValue(self, value, options):
        # Decode the incoming byte array to a string
        try:
            command = ''.join([chr(byte) for byte in value]).strip().lower()
            command = command.split(':')
        except UnicodeDecodeError:
            print('Received non-UTF-8 bytes')
            raise InvalidArgsException('Invalid encoding')

        if command[0]== 'start':
            self.handle_start(command[1])
        elif command[0] == 'stop':
            self.handle_stop()
        elif command[0] == 'calibrate':
            self.handle_calibrate()
        else:
            print('Unknown command received')
            raise InvalidArgsException('Invalid command')

    def handle_start(self, distance):
        print(f'Start command received with distance {distance}')
        # Optionally, notify the status
        process = subprocess.Popen(['python', 'poseDetection.py', str(distance)])
        status_chrc = self.service.get_characteristic_by_uuid(StatusCharacteristic.STATUS_CHRC_UUID)
        if status_chrc:
            status_chrc.update_status('Started')

    def handle_stop(self):
        print('Stop command received')
        # Optionally, notify the status
        pyautogui.press('q')
        status_chrc = self.service.get_characteristic_by_uuid(StatusCharacteristic.STATUS_CHRC_UUID)
        if status_chrc:
            status_chrc.update_status('Stopped')

    def handle_calibrate(self):
        print('Calibrate command received')
        pyautogui.press('c')
        # Optionally, notify the status
        status_chrc = self.service.get_characteristic_by_uuid(StatusCharacteristic.STATUS_CHRC_UUID)
        if status_chrc:
            status_chrc.update_status('Calibrated')


class StatusCharacteristic(Characteristic):
    """
    Notify characteristic to send status updates to the client.
    """
    STATUS_CHRC_UUID = '12345678-1234-5678-1234-56789abcdef2'  # Replace with your UUID

    def __init__(self, bus, index, service):
        Characteristic.__init__(
                self, bus, index,
                self.STATUS_CHRC_UUID,
                ['notify'],
                service)
        self.notifying = False

    def StartNotify(self):
        if self.notifying:
            print('Already notifying, nothing to do')
            return

        self.notifying = True
        print('Started notifying status changes')

    def StopNotify(self):
        if not self.notifying:
            print('Not notifying, nothing to do')
            return

        self.notifying = False
        print('Stopped notifying status changes')

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


# Adding helper method to Service to get characteristic by UUID
def service_get_characteristic_by_uuid(self, uuid):
    for chrc in self.characteristics:
        if chrc.uuid == uuid:
            return chrc
    return None

# Bind the helper method to Service class
Service.get_characteristic_by_uuid = service_get_characteristic_by_uuid


def register_app_cb():
    print('GATT application registered')


def register_app_error_cb(error):
    print('Failed to register application: ' + str(error))
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
    global process
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

    bus = dbus.SystemBus()

    adapter = find_adapter(bus)
    if not adapter:
        print('GattManager1 interface not found')
        return

    service_manager = dbus.Interface(
        bus.get_object(BLUEZ_SERVICE_NAME, adapter),
        GATT_MANAGER_IFACE)

    app = Application(bus)
    enable_advertising()
    mainloop = GLib.MainLoop()  # Updated MainLoop

    print('Registering GATT application...')

    service_manager.RegisterApplication(app.get_path(), {},
                                        reply_handler=register_app_cb,
                                        error_handler=register_app_error_cb)

    mainloop.run()


if __name__ == '__main__':
    main()
