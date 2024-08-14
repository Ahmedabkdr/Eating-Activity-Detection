/**
 *
 * @format
 * @flow strict-local
 */

import React, {useState, useEffect} from 'react';
import {
  StyleSheet,
  View,
  Text,
  Image,
  StatusBar,
  NativeModules,
  NativeEventEmitter,
  Platform,
  PermissionsAndroid,
  FlatList,
  TouchableHighlight,
  useColorScheme,
  Pressable,
  TouchableOpacity,
} from 'react-native';
import {RNS3} from 'react-native-aws3';
import {Amplify, Auth, Storage} from 'aws-amplify';
import SafeAreaView from 'react-native-safe-area-view';

import {Colors} from 'react-native/Libraries/NewAppScreen';

Amplify.configure({
  Auth: {
    identityPoolId: 'change', //Amazon Cognito Identity Pool ID
    region: 'eu-west-1', // REQUIRED - Amazon Cognito Region
  },
  Storage: {
    AWSS3: {
      bucket: 'eatingdetection', //Amazon S3 bucket name
      region: 'eu-west-1', //Amazon service region
    },
  },
});

const SECONDS_TO_SCAN_FOR = 3;
const SERVICE_UUIDS = [];
const ALLOW_DUPLICATES = false;

import BleManager from 'react-native-ble-manager';
import {Dirs, FileSystem} from 'react-native-file-access';
import RNFetchBlob from 'rn-fetch-blob';

import {NavigationContainer} from '@react-navigation/native';
import {createNativeStackNavigator} from '@react-navigation/native-stack';

const BleManagerModule = NativeModules.BleManager;
const bleManagerEmitter = new NativeEventEmitter(BleManagerModule);

const Stack = createNativeStackNavigator();

const App = () => {
  const [isScanning, setIsScanning] = useState(false);
  const [peripherals, setPeripherals] = useState(new Map());
  const [isConnected, setIsConnected] = useState(false);
  const [isEating, setIsEating] = useState(false);
  const [csvData, setCsvData] = useState('');
  const [showClassification, setShowClassification] = useState(false);
  const [csvModifiedDate, setCsvModifiedDate] = useState('');
  let [currentPeripheral, setCurrentPeripheral] = useState(null);
  const theme = useColorScheme();
  const fileName = 'accelerometer.csv'; //whatever you want to call your file
  const fileName2 = 'isEating.csv';
  const filePath = `${Dirs.DocumentDir}/${fileName}`;
  const filePath2 = `${Dirs.DocumentDir}/${fileName2}`;

  const updatePeripherals = (key, value) => {
    setPeripherals(new Map(peripherals.set(key, value)));
  };

  const startScan = () => {
    if (!isScanning) {
      try {
        console.log('Scanning...');
        setIsScanning(true);
        BleManager.scan(SERVICE_UUIDS, SECONDS_TO_SCAN_FOR, ALLOW_DUPLICATES);
      } catch (error) {
        console.error(error);
      }
    }
  };

  const handleStopScan = () => {
    setIsScanning(false);
    console.log('Scan is stopped');
  };

  const handleDisconnectedPeripheral = data => {
    let peripheral = peripherals.get(data.peripheral);
    if (peripheral) {
      currentPeripheral = null;
      peripheral.connected = false;
      updatePeripherals(peripheral.id, peripheral);
    }
    console.log('Disconnected from ' + data.peripheral);
  };

  const handleUpdateValueForCharacteristic = async data => {
    let measurements =
      calcAccel(data.value[1], data.value[0]) +
      ',' +
      calcAccel(data.value[3], data.value[2]) +
      ',' +
      calcAccel(data.value[5], data.value[4]);
    console.log(
      'Received data from ' +
        data.peripheral +
        ' characteristic ' +
        data.characteristic,
      measurements,
    );

    RNFetchBlob.fs.writeStream(filePath, 'utf8', true).then(stream => {
      stream.write(Date.now() + ',' + measurements + '\n');
      return stream.close();
    });
  };

  const calcAccel = (value1, value2) => {
    let accel = parseInt('0x' + convToHex(value1) + convToHex(value2));
    if ((accel & 0x8000) > 0) {
      accel = accel - 0x10000;
    }
    return accel / 1000;
  };

  const convToHex = value => {
    value = parseInt(value).toString(16).toUpperCase();
    if (value.length === 1) {
      value = '0' + value;
    }
    return value;
  };

  const handleDiscoverPeripheral = peripheral => {
    console.log('Got ble peripheral', peripheral);
    if (!peripheral.name) {
      peripheral.name = 'NO NAME';
    }
    updatePeripherals(peripheral.id, peripheral);
  };

  const togglePeripheralConnection = async peripheral => {
    if (peripheral && peripheral.connected) {
      BleManager.disconnect(peripheral.id);
    } else {
      connectPeripheral(peripheral);
    }
  };

  const toggleEating = () => {
    if (isEating) {
      setIsEating(false);
      RNFetchBlob.fs.writeStream(filePath2, 'utf8', true).then(stream => {
        stream.write(Date.now() + ',0' + '\n');
        return stream.close();
      });
    } else {
      setIsEating(true);
      RNFetchBlob.fs.writeStream(filePath2, 'utf8', true).then(stream => {
        stream.write(Date.now() + ',1' + '\n');
        return stream.close();
      });
    }
  };

  const toggleClassification = () => {
    if (showClassification) {
      setShowClassification(false);
    } else {
      setShowClassification(true);
      const csvKey = 'confusion_matrix.csv';

      Storage.get(csvKey)
        .then(url => {
          fetch(url)
            .then(response => response.text())
            .then(data => {
              setCsvData(data);

              Storage.get(csvKey, {download: true})
                .then(result => {
                  const lastModified = result.LastModified;
                  setCsvModifiedDate(lastModified.toString());
                })
                .catch(error => {
                  console.log('Error retrieving CSV metadata:', error);
                });
            })
            .catch(error => {
              console.log('Error retrieving CSV:', error);
            });
        })
        .catch(error => {
          console.log('Error retrieving CSV URL:', error);
        });
    }
  };

  const goToPeripheral = (peripheral, navigation) => {
    if (peripheral.connected) {
      navigation.navigate('Peripheral');
    } else {
      connectPeripheral(peripheral);
    }
  };

  const connectPeripheral = async peripheral => {
    try {
      if (peripheral) {
        markPeripheral({connecting: true});
        await BleManager.connect(peripheral.id);
        markPeripheral({connecting: false, connected: true});
        setIsConnected(true);
        setCurrentPeripheral(peripheral);

        await BleManager.retrieveServices(
          '4BE67751-5E54-3B46-2A20-E329813B524E',
        ).then(
          peripheralInfo => {
            // Success code
            console.log('Peripheral info:', peripheralInfo);
          },
          // "4BE67751-5E54-3B46-2A20-E329813B524E",
          // "E95D0753-251D-470A-A062-FA1922DFA9A8",
          // "E95DCA4B-251D-470A-A062-FA1922DFA9A8"
        );

        BleManager.startNotification(
          '4BE67751-5E54-3B46-2A20-E329813B524E',
          'E95D0753-251D-470A-A062-FA1922DFA9A8',
          'E95DCA4B-251D-470A-A062-FA1922DFA9A8',
        )
          .then(() => {
            // Success code
            console.log('Notification started');
          })
          .catch(error => {
            // Failure code
            console.log(error);
          });
      }
    } catch (error) {
      console.log('Connection error', error);
    }
    function markPeripheral(props) {
      updatePeripherals(peripheral.id, {...peripheral, ...props});
    }
  };

  useEffect(() => {
    BleManager.start({showAlert: false});
    const listeners = [
      bleManagerEmitter.addListener(
        'BleManagerDiscoverPeripheral',
        handleDiscoverPeripheral,
      ),
      bleManagerEmitter.addListener('BleManagerStopScan', handleStopScan),
      bleManagerEmitter.addListener(
        'BleManagerDisconnectPeripheral',
        handleDisconnectedPeripheral,
      ),
      bleManagerEmitter.addListener(
        'BleManagerDidUpdateValueForCharacteristic',
        handleUpdateValueForCharacteristic,
      ),
    ];

    handleAndroidPermissionCheck();

    return () => {
      console.log('unmount');
      for (const listener of listeners) {
        listener.remove();
      }
    };
  }, []);

  const handleAndroidPermissionCheck = () => {
    if (Platform.OS === 'android' && Platform.Version >= 23) {
      PermissionsAndroid.check(
        PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
      ).then(result => {
        if (result) {
          console.log('Permission is OK');
        } else {
          PermissionsAndroid.request(
            PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
          ).then(result => {
            if (result) {
              console.log('User accept');
            } else {
              console.log('User refuse');
            }
          });
        }
      });
    }
  };

  const renderItem = ({item, navigation}) => {
    const backgroundColor = item.connected ? '#069400' : Colors.white;
    return (
      <TouchableHighlight
        underlayColor="#0082FC"
        onPress={() => togglePeripheralConnection(item)}
        onLongPress={() => goToPeripheral(item, navigation)}>
        <View style={[styles.row, {backgroundColor}]}>
          <Text style={styles.peripheralName}>
            {item.name} {item.connecting && 'Connecting...'}
          </Text>
          <Text style={styles.rssi}>RSSI: {item.rssi}</Text>
          <Text style={styles.peripheralId}>{item.id}</Text>
        </View>
      </TouchableHighlight>
    );
  };

  const uploadData = () => {
    const file = {
      uri: filePath,
      name: fileName,
      type: 'text/csv',
    };

    const file2 = {
      uri: filePath2,
      name: fileName2,
      type: 'text/csv',
    };

    console.log(file);
    const config = {
      keyPrefix: '',
      bucket: 'newacceldata',
      region: 'eu-west-1',
      accessKey: 'accesskey',
      secretKey: 'secretkey',
      successActionStatus: 201,
    };

    RNS3.put(file, config).then(response => {
      console.log(response);
    });

    RNS3.put(file2, config).then(response => {
      console.log(response);
    });
  };

  const classify = () => {
    const file = {
      uri: filePath,
      name: fileName,
      type: 'text/csv',
    };

    const file2 = {
      uri: filePath2,
      name: fileName2,
      type: 'text/csv',
    };

    console.log(file);
    const config = {
      keyPrefix: '',
      bucket: 'eatingdetection',
      region: 'eu-west-1',
      accessKey: 'accesskey',
      secretKey: 'secretkey',
      successActionStatus: 201,
    };

    RNS3.put(file, config).then(response => {
      console.log(response);
    });

    RNS3.put(file2, config).then(response => {
      console.log(response);
    });
  };

  const HomeScreen = ({navigation}) => {
    // useEffect(() => {
    //   if (isConnected) {
    //     navigation.navigate('Peripheral')
    //   }
    // })

    return (
      <>
        <StatusBar />
        <SafeAreaView
          style={styles.body}
          forceInset={{top: 'always', bottom: 'always'}}>
          <Pressable style={styles.scanButton} onPress={startScan}>
            <Text style={styles.scanButtonText}>
              {isScanning ? 'Scanning...' : 'Scan Bluetooth'}
            </Text>
          </Pressable>

          {Array.from(peripherals.values()).length == 0 && (
            <View style={styles.row}>
              <Text style={styles.noPeripherals}>
                No Peripherals, press "Scan Bluetooth" above
              </Text>
            </View>
          )}
          <FlatList
            data={Array.from(peripherals.values())}
            contentContainerStyle={{rowGap: 12}}
            renderItem={({item}) => renderItem({item, navigation})}
            keyExtractor={item => item.id}
          />
        </SafeAreaView>
      </>
    );
  };

  const PeripheralScreen = ({navigation}) => {
    const backgroundColor = Colors.white;

    // @ts-ignore
    return (
      <>
        <StatusBar />
        <SafeAreaView
          style={styles.body}
          forceInset={{top: 'always', bottom: 'always'}}>
          <TouchableOpacity
            style={styles.scanButton}
            onPress={() => navigation.navigate('Home')}>
            <Text style={styles.scanButtonText}>Back</Text>
          </TouchableOpacity>
          <View style={[styles.row, {backgroundColor}]}>
            <Text style={styles.rssi}>RSSI: {currentPeripheral.rssi}</Text>
            <Text style={styles.peripheralId}>{currentPeripheral.id}</Text>
          </View>
          <TouchableOpacity
            style={styles.scanButton}
            onPress={() => toggleEating()}>
            <Text style={styles.scanButtonText}>
              {isEating ? 'Stop Eating' : 'Start Eating'}
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.scanButton}
            onPress={() => uploadData()}>
            <Text style={styles.scanButtonText}>Upload Accelerometer Data</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.scanButton}
            onPress={() => classify()}>
            <Text style={styles.scanButtonText}>Classify Eating</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.scanButton}
            onPress={() => toggleClassification()}>
            <Text style={styles.scanButtonText}>
              {showClassification
                ? 'Hide Classification'
                : 'Show Classification'}
            </Text>
          </TouchableOpacity>
          <View>
            {showClassification ? (
              csvData ? (
                <View>
                  {csvData.split('\n\n').map((matrix, index) => {
                    const [topRow, bottomRow] = matrix.split('\n'); // Split into top and bottom rows

                    const [tn, fp] = topRow.split(',').map(parseFloat); // Assuming the format is: "TN, FP"
                    const [fn, tp] = bottomRow.split(',').map(parseFloat); // Assuming the format is: "FN, TP"

                    const perClassAccuracies = [];
                    let overallAccuracy = 0;

                    perClassAccuracies.push(tn / (tn + fp)).toFixed(3);
                    perClassAccuracies.push(tp / (tp + fn)).toFixed(3);
                    overallAccuracy = ((tp + tn) / (tp + tn + fp + fn)).toFixed(3);

                    const allPositives = tp + fp;
                    const precision = parseFloat(
                      (tp / allPositives).toFixed(3),
                    );
                    const recall = parseFloat((tp / (tp + fn)).toFixed(3));
                    const fScore = (
                      (2 * precision * recall) /
                      (precision + recall)
                    ).toFixed(3);
                    return (
                      <View key={index}>
                        <Text style={[styles.scanButtonText, {marginLeft: 20}]}>
                          True Negatives: {tn}
                        </Text>

                        <Text style={[styles.scanButtonText, {marginLeft: 20}]}>
                          False Positives: {fp}
                        </Text>

                        <Text style={[styles.scanButtonText, {marginLeft: 20}]}>
                          False Negatives: {fn}
                        </Text>

                        <Text style={[styles.scanButtonText, {marginLeft: 20}]}>
                          True Positives: {tp}{'\n'}
                        </Text>
                        <Text style={[styles.scanButtonText, {marginLeft: 20}]}>
                          Per-class Accuracy:{' '}
                          {perClassAccuracies
                            .map(accuracy => accuracy.toFixed(3))
                            .join(', ')}
                        </Text>

                        <Text style={[styles.scanButtonText, {marginLeft: 20}]}>
                          Overall Accuracy: {overallAccuracy}
                        </Text>

                        <Text style={[styles.scanButtonText, {marginLeft: 20}]}>
                          Precision: {precision}
                        </Text>

                        <Text style={[styles.scanButtonText, {marginLeft: 20}]}>
                          Recall: {recall}
                        </Text>

                        <Text style={[styles.scanButtonText, {marginLeft: 20}]}>
                          F-score: {fScore}
                        </Text>
                      </View>
                    );
                  })}
                  <Text style={[styles.scanButtonText, {textAlign: 'center'}]}>
                    {'\n\n'} Predictions made at: {csvModifiedDate}
                  </Text>
                </View>
              ) : (
                <Text>Loading...</Text>
              )
            ) : null}
          </View>
        </SafeAreaView>
      </>
    );
  };

  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen
          name="Home"
          component={HomeScreen}
          options={{headerShown: false}}
        />
        <Stack.Screen
          name="Peripheral"
          component={PeripheralScreen}
          options={{headerShown: false}}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

const boxShadow = {
  shadowColor: '#000',
  shadowOffset: {
    width: 0,
    height: 2,
  },
  shadowOpacity: 0.25,
  shadowRadius: 3.84,
  elevation: 5,
};

const styles = StyleSheet.create({
  engine: {
    position: 'absolute',
    right: 10,
    bottom: 0,
    color: Colors.black,
  },
  scanButton: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    backgroundColor: '#0a398a',
    margin: 10,
    borderRadius: 12,
    ...boxShadow,
  },
  scanButtonText: {
    fontSize: 20,
    letterSpacing: 0.25,
    color: Colors.white,
  },
  body: {
    backgroundColor: '#0082FC',
    flex: 1,
  },
  sectionContainer: {
    marginTop: 32,
    paddingHorizontal: 24,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: '600',
    color: Colors.black,
  },
  sectionDescription: {
    marginTop: 8,
    fontSize: 18,
    fontWeight: '400',
    color: Colors.dark,
  },
  highlight: {
    fontWeight: '700',
  },
  footer: {
    color: Colors.dark,
    fontSize: 12,
    fontWeight: '600',
    padding: 4,
    paddingRight: 12,
    textAlign: 'right',
  },
  peripheralName: {
    fontSize: 16,
    textAlign: 'center',
    padding: 10,
  },
  rssi: {
    fontSize: 12,
    textAlign: 'center',
    padding: 2,
  },
  peripheralId: {
    fontSize: 12,
    textAlign: 'center',
    padding: 2,
    paddingBottom: 20,
  },
  row: {
    marginLeft: 10,
    marginRight: 10,
    borderRadius: 20,
    ...boxShadow,
  },
  noPeripherals: {
    margin: 10,
    textAlign: 'center',
    color: Colors.white,
  },
});

export default App;
