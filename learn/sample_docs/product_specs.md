# Halcyon Robotics — Product Specifications

Halcyon builds autonomous mobile robots for warehouses. There are two models in the current line-up.

## Kestrel K2 (heavy-duty)

The Kestrel K2 moves pallets and large totes between storage and packing stations.

- Payload: 45 kg
- Robot weight: 68 kg
- Battery life: 6.5 hours of continuous operation
- Charging: 0 to 80% in 50 minutes on the docking station
- Maximum speed: 2.2 m/s (limited to 1.2 m/s near people)
- Sensors: 360° lidar, two depth cameras, bumper strip
- Ingress protection: IP54
- Current firmware: 4.3

## Wren W1 (light-duty)

The Wren W1 carries single totes and parcels and is designed for narrow aisles.

- Payload: 12 kg
- Robot weight: 21 kg
- Battery life: 9 hours of continuous operation
- Charging: 0 to 80% in 35 minutes
- Maximum speed: 1.8 m/s
- Sensors: front lidar, one depth camera
- Ingress protection: IP42
- Current firmware: 2.7

## Fleet limits

A single fleet manager instance supports up to 250 robots. Mixed fleets of Kestrel and Wren robots are supported from fleet manager version 3.0.

## Common error codes

- E-104: localisation lost. The robot stops and waits for an operator to confirm its position on the map.
- E-217: wheel motor overcurrent. Power-cycle the robot; if the error returns within 24 hours, replace the drive module.
- E-330: battery temperature too high. The robot returns to its dock and stops charging until the battery cools below 40 °C.
- E-512: emergency stop pressed. Release the button and resume the mission from the fleet manager.

## Warranty

Both models come with a 3-year warranty on the chassis and drive train and a 2-year warranty on batteries.
