/usr/bin/python3.5 /home/qtz/program/tensorflow/tensorflow-learning/CNN/tf02_advancedcnn.py
>> Downloading cifar-10-binary.tar.gz 100.0%
Successfully downloaded cifar-10-binary.tar.gz 170052171 bytes.
Filling queue with 20000 CIFAR images before starting to train. This will take a few minutes.
2017-06-27 13:30:02.112591: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-27 13:30:02.112610: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-27 13:30:02.112615: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-06-27 13:30:02.112618: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-27 13:30:02.112621: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2017-06-27 13:30:02.431811: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2017-06-27 13:30:02.432183: I tensorflow/core/common_runtime/gpu/gpu_device.cc:887] Found device 0 with properties: 
name: GeForce 940MX
major: 5 minor: 0 memoryClockRate (GHz) 1.2415
pciBusID 0000:02:00.0
Total memory: 1.96GiB
Free memory: 1.26GiB
2017-06-27 13:30:02.432198: I tensorflow/core/common_runtime/gpu/gpu_device.cc:908] DMA: 0 
2017-06-27 13:30:02.432203: I tensorflow/core/common_runtime/gpu/gpu_device.cc:918] 0:   Y 
2017-06-27 13:30:02.432211: I tensorflow/core/common_runtime/gpu/gpu_device.cc:977] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce 940MX, pci bus id: 0000:02:00.0)
step 0,loss=4.67 (7.8 examples/sec; 16.480 sec/batch)
step 10,loss=3.78 (717.1 examples/sec; 0.179 sec/batch)
step 20,loss=3.28 (725.7 examples/sec; 0.176 sec/batch)
step 30,loss=2.90 (747.7 examples/sec; 0.171 sec/batch)
step 40,loss=2.65 (764.5 examples/sec; 0.167 sec/batch)
step 50,loss=2.56 (690.1 examples/sec; 0.185 sec/batch)
step 60,loss=2.33 (731.9 examples/sec; 0.175 sec/batch)
step 70,loss=2.26 (721.7 examples/sec; 0.177 sec/batch)
step 80,loss=2.28 (713.9 examples/sec; 0.179 sec/batch)
step 90,loss=2.15 (748.7 examples/sec; 0.171 sec/batch)
step 100,loss=2.28 (693.1 examples/sec; 0.185 sec/batch)
step 110,loss=2.17 (642.4 examples/sec; 0.199 sec/batch)
step 120,loss=2.09 (765.1 examples/sec; 0.167 sec/batch)
step 130,loss=2.14 (697.3 examples/sec; 0.184 sec/batch)
step 140,loss=2.10 (759.4 examples/sec; 0.169 sec/batch)
step 150,loss=2.05 (744.5 examples/sec; 0.172 sec/batch)
step 160,loss=2.08 (755.7 examples/sec; 0.169 sec/batch)
step 170,loss=2.07 (757.4 examples/sec; 0.169 sec/batch)
step 180,loss=1.99 (711.3 examples/sec; 0.180 sec/batch)
step 190,loss=1.99 (766.4 examples/sec; 0.167 sec/batch)
step 200,loss=2.12 (776.3 examples/sec; 0.165 sec/batch)
step 210,loss=2.04 (699.8 examples/sec; 0.183 sec/batch)
step 220,loss=1.97 (695.6 examples/sec; 0.184 sec/batch)
step 230,loss=2.05 (629.0 examples/sec; 0.203 sec/batch)
step 240,loss=1.99 (766.4 examples/sec; 0.167 sec/batch)
step 250,loss=2.11 (656.6 examples/sec; 0.195 sec/batch)
step 260,loss=1.99 (686.7 examples/sec; 0.186 sec/batch)
step 270,loss=1.92 (612.4 examples/sec; 0.209 sec/batch)
step 280,loss=2.07 (731.4 examples/sec; 0.175 sec/batch)
step 290,loss=1.78 (759.5 examples/sec; 0.169 sec/batch)
step 300,loss=1.90 (665.1 examples/sec; 0.192 sec/batch)
step 310,loss=1.99 (730.2 examples/sec; 0.175 sec/batch)
step 320,loss=1.93 (632.8 examples/sec; 0.202 sec/batch)
step 330,loss=2.09 (719.8 examples/sec; 0.178 sec/batch)
step 340,loss=1.93 (756.9 examples/sec; 0.169 sec/batch)
step 350,loss=1.99 (742.6 examples/sec; 0.172 sec/batch)
step 360,loss=2.14 (751.6 examples/sec; 0.170 sec/batch)
step 370,loss=2.09 (757.9 examples/sec; 0.169 sec/batch)
step 380,loss=2.01 (736.8 examples/sec; 0.174 sec/batch)
step 390,loss=1.91 (777.6 examples/sec; 0.165 sec/batch)
step 400,loss=1.87 (703.6 examples/sec; 0.182 sec/batch)
step 410,loss=2.01 (753.2 examples/sec; 0.170 sec/batch)
step 420,loss=1.87 (758.4 examples/sec; 0.169 sec/batch)
step 430,loss=1.96 (748.3 examples/sec; 0.171 sec/batch)
step 440,loss=1.72 (733.3 examples/sec; 0.175 sec/batch)
step 450,loss=1.92 (789.4 examples/sec; 0.162 sec/batch)
step 460,loss=1.91 (749.0 examples/sec; 0.171 sec/batch)
step 470,loss=1.97 (757.6 examples/sec; 0.169 sec/batch)
step 480,loss=1.93 (665.0 examples/sec; 0.192 sec/batch)
step 490,loss=1.91 (602.2 examples/sec; 0.213 sec/batch)
step 500,loss=1.76 (716.6 examples/sec; 0.179 sec/batch)
step 510,loss=1.90 (696.5 examples/sec; 0.184 sec/batch)
step 520,loss=1.77 (726.3 examples/sec; 0.176 sec/batch)
step 530,loss=1.83 (720.9 examples/sec; 0.178 sec/batch)
step 540,loss=1.97 (738.2 examples/sec; 0.173 sec/batch)
step 550,loss=1.90 (607.3 examples/sec; 0.211 sec/batch)
step 560,loss=1.78 (778.8 examples/sec; 0.164 sec/batch)
step 570,loss=1.91 (743.1 examples/sec; 0.172 sec/batch)
step 580,loss=1.89 (736.9 examples/sec; 0.174 sec/batch)
step 590,loss=1.81 (697.7 examples/sec; 0.183 sec/batch)
step 600,loss=1.91 (675.0 examples/sec; 0.190 sec/batch)
step 610,loss=1.67 (712.1 examples/sec; 0.180 sec/batch)
step 620,loss=1.87 (760.5 examples/sec; 0.168 sec/batch)
step 630,loss=1.94 (698.5 examples/sec; 0.183 sec/batch)
step 640,loss=1.75 (725.7 examples/sec; 0.176 sec/batch)
step 650,loss=1.78 (738.8 examples/sec; 0.173 sec/batch)
step 660,loss=1.78 (737.5 examples/sec; 0.174 sec/batch)
step 670,loss=1.81 (695.0 examples/sec; 0.184 sec/batch)
step 680,loss=1.93 (747.0 examples/sec; 0.171 sec/batch)
step 690,loss=1.88 (738.4 examples/sec; 0.173 sec/batch)
step 700,loss=1.90 (748.8 examples/sec; 0.171 sec/batch)
step 710,loss=1.87 (746.4 examples/sec; 0.171 sec/batch)
step 720,loss=1.70 (653.2 examples/sec; 0.196 sec/batch)
step 730,loss=1.83 (720.1 examples/sec; 0.178 sec/batch)
step 740,loss=1.70 (587.3 examples/sec; 0.218 sec/batch)
step 750,loss=1.83 (742.4 examples/sec; 0.172 sec/batch)
step 760,loss=1.76 (753.6 examples/sec; 0.170 sec/batch)
step 770,loss=1.86 (730.4 examples/sec; 0.175 sec/batch)
step 780,loss=1.76 (745.4 examples/sec; 0.172 sec/batch)
step 790,loss=1.95 (740.7 examples/sec; 0.173 sec/batch)
step 800,loss=1.77 (746.6 examples/sec; 0.171 sec/batch)
step 810,loss=1.93 (644.3 examples/sec; 0.199 sec/batch)
step 820,loss=1.68 (682.7 examples/sec; 0.187 sec/batch)
step 830,loss=1.83 (633.6 examples/sec; 0.202 sec/batch)
step 840,loss=1.75 (770.8 examples/sec; 0.166 sec/batch)
step 850,loss=1.87 (677.2 examples/sec; 0.189 sec/batch)
step 860,loss=1.67 (686.1 examples/sec; 0.187 sec/batch)
step 870,loss=1.77 (665.1 examples/sec; 0.192 sec/batch)
step 880,loss=1.59 (726.5 examples/sec; 0.176 sec/batch)
step 890,loss=1.79 (742.0 examples/sec; 0.173 sec/batch)
step 900,loss=1.92 (709.9 examples/sec; 0.180 sec/batch)
step 910,loss=1.70 (641.2 examples/sec; 0.200 sec/batch)
step 920,loss=1.63 (706.4 examples/sec; 0.181 sec/batch)
step 930,loss=1.73 (665.5 examples/sec; 0.192 sec/batch)
step 940,loss=1.85 (679.3 examples/sec; 0.188 sec/batch)
step 950,loss=1.87 (667.6 examples/sec; 0.192 sec/batch)
step 960,loss=1.90 (752.7 examples/sec; 0.170 sec/batch)
step 970,loss=1.65 (746.4 examples/sec; 0.171 sec/batch)
step 980,loss=1.82 (730.7 examples/sec; 0.175 sec/batch)
step 990,loss=1.94 (696.6 examples/sec; 0.184 sec/batch)
step 1000,loss=1.85 (753.0 examples/sec; 0.170 sec/batch)
step 1010,loss=1.79 (734.6 examples/sec; 0.174 sec/batch)
step 1020,loss=1.67 (750.8 examples/sec; 0.170 sec/batch)
step 1030,loss=1.80 (700.2 examples/sec; 0.183 sec/batch)
step 1040,loss=1.69 (759.0 examples/sec; 0.169 sec/batch)
step 1050,loss=1.78 (704.5 examples/sec; 0.182 sec/batch)
step 1060,loss=1.81 (690.7 examples/sec; 0.185 sec/batch)
step 1070,loss=1.79 (643.3 examples/sec; 0.199 sec/batch)
step 1080,loss=1.69 (737.7 examples/sec; 0.174 sec/batch)
step 1090,loss=1.88 (675.0 examples/sec; 0.190 sec/batch)
step 1100,loss=1.64 (739.7 examples/sec; 0.173 sec/batch)
step 1110,loss=1.93 (749.0 examples/sec; 0.171 sec/batch)
step 1120,loss=1.65 (708.2 examples/sec; 0.181 sec/batch)
step 1130,loss=1.77 (751.3 examples/sec; 0.170 sec/batch)
step 1140,loss=1.66 (638.9 examples/sec; 0.200 sec/batch)
step 1150,loss=1.66 (671.7 examples/sec; 0.191 sec/batch)
step 1160,loss=1.84 (727.9 examples/sec; 0.176 sec/batch)
step 1170,loss=1.74 (738.7 examples/sec; 0.173 sec/batch)
step 1180,loss=1.74 (668.1 examples/sec; 0.192 sec/batch)
step 1190,loss=1.81 (615.9 examples/sec; 0.208 sec/batch)
step 1200,loss=1.90 (752.4 examples/sec; 0.170 sec/batch)
step 1210,loss=1.72 (687.9 examples/sec; 0.186 sec/batch)
step 1220,loss=1.75 (688.3 examples/sec; 0.186 sec/batch)
step 1230,loss=1.83 (667.2 examples/sec; 0.192 sec/batch)
step 1240,loss=1.78 (712.9 examples/sec; 0.180 sec/batch)
step 1250,loss=1.59 (746.1 examples/sec; 0.172 sec/batch)
step 1260,loss=1.78 (745.1 examples/sec; 0.172 sec/batch)
step 1270,loss=1.80 (734.0 examples/sec; 0.174 sec/batch)
step 1280,loss=1.82 (753.7 examples/sec; 0.170 sec/batch)
step 1290,loss=1.96 (717.5 examples/sec; 0.178 sec/batch)
step 1300,loss=1.81 (648.3 examples/sec; 0.197 sec/batch)
step 1310,loss=1.81 (754.4 examples/sec; 0.170 sec/batch)
step 1320,loss=1.66 (730.3 examples/sec; 0.175 sec/batch)
step 1330,loss=1.83 (757.6 examples/sec; 0.169 sec/batch)
step 1340,loss=1.61 (772.6 examples/sec; 0.166 sec/batch)
step 1350,loss=1.75 (736.8 examples/sec; 0.174 sec/batch)
step 1360,loss=1.83 (673.7 examples/sec; 0.190 sec/batch)
step 1370,loss=1.80 (715.6 examples/sec; 0.179 sec/batch)
step 1380,loss=1.79 (747.9 examples/sec; 0.171 sec/batch)
step 1390,loss=1.72 (657.2 examples/sec; 0.195 sec/batch)
step 1400,loss=1.69 (731.9 examples/sec; 0.175 sec/batch)
step 1410,loss=1.67 (757.3 examples/sec; 0.169 sec/batch)
step 1420,loss=1.85 (708.5 examples/sec; 0.181 sec/batch)
step 1430,loss=1.87 (589.5 examples/sec; 0.217 sec/batch)
step 1440,loss=1.68 (702.8 examples/sec; 0.182 sec/batch)
step 1450,loss=1.78 (739.5 examples/sec; 0.173 sec/batch)
step 1460,loss=1.69 (712.0 examples/sec; 0.180 sec/batch)
step 1470,loss=1.58 (758.0 examples/sec; 0.169 sec/batch)
step 1480,loss=1.63 (745.2 examples/sec; 0.172 sec/batch)
step 1490,loss=1.59 (724.2 examples/sec; 0.177 sec/batch)
step 1500,loss=1.77 (714.3 examples/sec; 0.179 sec/batch)
step 1510,loss=1.59 (758.5 examples/sec; 0.169 sec/batch)
step 1520,loss=1.73 (725.0 examples/sec; 0.177 sec/batch)
step 1530,loss=1.77 (725.5 examples/sec; 0.176 sec/batch)
step 1540,loss=1.56 (714.4 examples/sec; 0.179 sec/batch)
step 1550,loss=1.60 (738.2 examples/sec; 0.173 sec/batch)
step 1560,loss=1.65 (745.8 examples/sec; 0.172 sec/batch)
step 1570,loss=1.73 (746.3 examples/sec; 0.172 sec/batch)
step 1580,loss=1.68 (703.7 examples/sec; 0.182 sec/batch)
step 1590,loss=1.66 (702.4 examples/sec; 0.182 sec/batch)
step 1600,loss=1.67 (763.8 examples/sec; 0.168 sec/batch)
step 1610,loss=1.65 (719.3 examples/sec; 0.178 sec/batch)
step 1620,loss=1.68 (697.0 examples/sec; 0.184 sec/batch)
step 1630,loss=1.77 (634.3 examples/sec; 0.202 sec/batch)
step 1640,loss=1.61 (727.5 examples/sec; 0.176 sec/batch)
step 1650,loss=1.50 (732.5 examples/sec; 0.175 sec/batch)
step 1660,loss=1.73 (624.5 examples/sec; 0.205 sec/batch)
step 1670,loss=1.62 (671.4 examples/sec; 0.191 sec/batch)
step 1680,loss=1.93 (665.0 examples/sec; 0.192 sec/batch)
step 1690,loss=1.79 (560.2 examples/sec; 0.228 sec/batch)
step 1700,loss=1.81 (671.9 examples/sec; 0.190 sec/batch)
step 1710,loss=1.75 (715.6 examples/sec; 0.179 sec/batch)
step 1720,loss=1.77 (727.4 examples/sec; 0.176 sec/batch)
step 1730,loss=1.81 (748.2 examples/sec; 0.171 sec/batch)
step 1740,loss=1.63 (740.8 examples/sec; 0.173 sec/batch)
step 1750,loss=1.58 (730.2 examples/sec; 0.175 sec/batch)
step 1760,loss=1.61 (754.6 examples/sec; 0.170 sec/batch)
step 1770,loss=1.63 (738.1 examples/sec; 0.173 sec/batch)
step 1780,loss=1.76 (711.5 examples/sec; 0.180 sec/batch)
step 1790,loss=1.73 (725.1 examples/sec; 0.177 sec/batch)
step 1800,loss=1.76 (682.2 examples/sec; 0.188 sec/batch)
step 1810,loss=1.65 (672.3 examples/sec; 0.190 sec/batch)
step 1820,loss=1.85 (689.5 examples/sec; 0.186 sec/batch)
step 1830,loss=1.78 (693.3 examples/sec; 0.185 sec/batch)
step 1840,loss=1.69 (701.5 examples/sec; 0.182 sec/batch)
step 1850,loss=1.56 (677.2 examples/sec; 0.189 sec/batch)
step 1860,loss=1.81 (712.0 examples/sec; 0.180 sec/batch)
step 1870,loss=1.70 (719.0 examples/sec; 0.178 sec/batch)
step 1880,loss=1.75 (773.8 examples/sec; 0.165 sec/batch)
step 1890,loss=1.71 (759.8 examples/sec; 0.168 sec/batch)
step 1900,loss=1.56 (745.3 examples/sec; 0.172 sec/batch)
step 1910,loss=1.93 (647.4 examples/sec; 0.198 sec/batch)
step 1920,loss=1.65 (686.7 examples/sec; 0.186 sec/batch)
step 1930,loss=1.65 (712.7 examples/sec; 0.180 sec/batch)
step 1940,loss=1.63 (591.8 examples/sec; 0.216 sec/batch)
step 1950,loss=1.65 (679.8 examples/sec; 0.188 sec/batch)
step 1960,loss=1.75 (648.6 examples/sec; 0.197 sec/batch)
step 1970,loss=1.48 (705.9 examples/sec; 0.181 sec/batch)
step 1980,loss=1.58 (743.1 examples/sec; 0.172 sec/batch)
step 1990,loss=1.74 (724.5 examples/sec; 0.177 sec/batch)
step 2000,loss=1.62 (747.6 examples/sec; 0.171 sec/batch)
step 2010,loss=1.67 (704.9 examples/sec; 0.182 sec/batch)
step 2020,loss=1.62 (657.7 examples/sec; 0.195 sec/batch)
step 2030,loss=1.76 (659.6 examples/sec; 0.194 sec/batch)
step 2040,loss=1.82 (648.9 examples/sec; 0.197 sec/batch)
step 2050,loss=1.81 (575.4 examples/sec; 0.222 sec/batch)
step 2060,loss=1.53 (698.2 examples/sec; 0.183 sec/batch)
step 2070,loss=1.57 (757.0 examples/sec; 0.169 sec/batch)
step 2080,loss=1.64 (670.6 examples/sec; 0.191 sec/batch)
step 2090,loss=1.77 (652.0 examples/sec; 0.196 sec/batch)
step 2100,loss=1.61 (696.7 examples/sec; 0.184 sec/batch)
step 2110,loss=1.79 (644.8 examples/sec; 0.199 sec/batch)
step 2120,loss=1.72 (683.5 examples/sec; 0.187 sec/batch)
step 2130,loss=1.74 (554.2 examples/sec; 0.231 sec/batch)
step 2140,loss=1.68 (601.5 examples/sec; 0.213 sec/batch)
step 2150,loss=1.73 (694.0 examples/sec; 0.184 sec/batch)
step 2160,loss=1.63 (693.0 examples/sec; 0.185 sec/batch)
step 2170,loss=1.69 (692.4 examples/sec; 0.185 sec/batch)
step 2180,loss=1.71 (699.8 examples/sec; 0.183 sec/batch)
step 2190,loss=1.75 (712.0 examples/sec; 0.180 sec/batch)
step 2200,loss=1.78 (726.1 examples/sec; 0.176 sec/batch)
step 2210,loss=1.78 (719.1 examples/sec; 0.178 sec/batch)
step 2220,loss=1.64 (679.6 examples/sec; 0.188 sec/batch)
step 2230,loss=1.59 (702.2 examples/sec; 0.182 sec/batch)
step 2240,loss=1.78 (702.3 examples/sec; 0.182 sec/batch)
step 2250,loss=1.73 (719.8 examples/sec; 0.178 sec/batch)
step 2260,loss=1.73 (727.5 examples/sec; 0.176 sec/batch)
step 2270,loss=1.68 (674.6 examples/sec; 0.190 sec/batch)
step 2280,loss=1.58 (690.0 examples/sec; 0.186 sec/batch)
step 2290,loss=1.74 (671.9 examples/sec; 0.191 sec/batch)
step 2300,loss=1.66 (706.8 examples/sec; 0.181 sec/batch)
step 2310,loss=1.43 (679.5 examples/sec; 0.188 sec/batch)
step 2320,loss=1.57 (723.9 examples/sec; 0.177 sec/batch)
step 2330,loss=1.70 (675.6 examples/sec; 0.189 sec/batch)
step 2340,loss=1.65 (607.7 examples/sec; 0.211 sec/batch)
step 2350,loss=1.67 (657.5 examples/sec; 0.195 sec/batch)
step 2360,loss=1.73 (696.3 examples/sec; 0.184 sec/batch)
step 2370,loss=1.69 (721.5 examples/sec; 0.177 sec/batch)
step 2380,loss=1.61 (690.6 examples/sec; 0.185 sec/batch)
step 2390,loss=1.77 (745.4 examples/sec; 0.172 sec/batch)
step 2400,loss=1.61 (681.1 examples/sec; 0.188 sec/batch)
step 2410,loss=1.60 (662.6 examples/sec; 0.193 sec/batch)
step 2420,loss=1.51 (638.5 examples/sec; 0.200 sec/batch)
step 2430,loss=1.76 (729.2 examples/sec; 0.176 sec/batch)
step 2440,loss=1.69 (730.4 examples/sec; 0.175 sec/batch)
step 2450,loss=1.68 (701.4 examples/sec; 0.182 sec/batch)
step 2460,loss=1.61 (719.6 examples/sec; 0.178 sec/batch)
step 2470,loss=1.77 (731.2 examples/sec; 0.175 sec/batch)
step 2480,loss=1.76 (706.7 examples/sec; 0.181 sec/batch)
step 2490,loss=1.61 (744.2 examples/sec; 0.172 sec/batch)
step 2500,loss=1.58 (753.4 examples/sec; 0.170 sec/batch)
step 2510,loss=1.67 (762.4 examples/sec; 0.168 sec/batch)
step 2520,loss=1.68 (670.6 examples/sec; 0.191 sec/batch)
step 2530,loss=1.59 (750.1 examples/sec; 0.171 sec/batch)
step 2540,loss=1.64 (717.8 examples/sec; 0.178 sec/batch)
step 2550,loss=1.59 (759.6 examples/sec; 0.169 sec/batch)
step 2560,loss=1.62 (718.2 examples/sec; 0.178 sec/batch)
step 2570,loss=1.78 (623.5 examples/sec; 0.205 sec/batch)
step 2580,loss=1.64 (716.3 examples/sec; 0.179 sec/batch)
step 2590,loss=1.70 (566.0 examples/sec; 0.226 sec/batch)
step 2600,loss=1.41 (735.4 examples/sec; 0.174 sec/batch)
step 2610,loss=1.62 (689.4 examples/sec; 0.186 sec/batch)
step 2620,loss=1.63 (665.0 examples/sec; 0.192 sec/batch)
step 2630,loss=1.52 (683.7 examples/sec; 0.187 sec/batch)
step 2640,loss=1.76 (629.8 examples/sec; 0.203 sec/batch)
step 2650,loss=1.66 (704.7 examples/sec; 0.182 sec/batch)
step 2660,loss=1.92 (677.1 examples/sec; 0.189 sec/batch)
step 2670,loss=1.75 (740.7 examples/sec; 0.173 sec/batch)
step 2680,loss=1.63 (735.3 examples/sec; 0.174 sec/batch)
step 2690,loss=1.54 (745.5 examples/sec; 0.172 sec/batch)
step 2700,loss=1.71 (677.1 examples/sec; 0.189 sec/batch)
step 2710,loss=1.61 (754.1 examples/sec; 0.170 sec/batch)
step 2720,loss=1.62 (756.9 examples/sec; 0.169 sec/batch)
step 2730,loss=1.74 (735.4 examples/sec; 0.174 sec/batch)
step 2740,loss=1.53 (733.4 examples/sec; 0.175 sec/batch)
step 2750,loss=1.81 (691.9 examples/sec; 0.185 sec/batch)
step 2760,loss=1.78 (624.6 examples/sec; 0.205 sec/batch)
step 2770,loss=1.63 (649.1 examples/sec; 0.197 sec/batch)
step 2780,loss=1.64 (708.5 examples/sec; 0.181 sec/batch)
step 2790,loss=1.70 (696.7 examples/sec; 0.184 sec/batch)
step 2800,loss=1.85 (744.9 examples/sec; 0.172 sec/batch)
step 2810,loss=1.60 (774.4 examples/sec; 0.165 sec/batch)
step 2820,loss=1.60 (701.4 examples/sec; 0.182 sec/batch)
step 2830,loss=1.66 (721.8 examples/sec; 0.177 sec/batch)
step 2840,loss=1.75 (746.8 examples/sec; 0.171 sec/batch)
step 2850,loss=1.72 (730.4 examples/sec; 0.175 sec/batch)
step 2860,loss=1.67 (760.0 examples/sec; 0.168 sec/batch)
step 2870,loss=1.49 (741.9 examples/sec; 0.173 sec/batch)
step 2880,loss=1.55 (421.0 examples/sec; 0.304 sec/batch)
step 2890,loss=1.76 (705.7 examples/sec; 0.181 sec/batch)
step 2900,loss=1.74 (689.6 examples/sec; 0.186 sec/batch)
step 2910,loss=1.85 (718.8 examples/sec; 0.178 sec/batch)
step 2920,loss=1.71 (759.7 examples/sec; 0.168 sec/batch)
step 2930,loss=1.57 (732.4 examples/sec; 0.175 sec/batch)
step 2940,loss=1.53 (728.3 examples/sec; 0.176 sec/batch)
step 2950,loss=1.56 (740.4 examples/sec; 0.173 sec/batch)
step 2960,loss=1.74 (711.4 examples/sec; 0.180 sec/batch)
step 2970,loss=1.60 (741.4 examples/sec; 0.173 sec/batch)
step 2980,loss=1.74 (731.1 examples/sec; 0.175 sec/batch)
step 2990,loss=1.51 (743.4 examples/sec; 0.172 sec/batch)
precision @ 1 = 0.618

Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)