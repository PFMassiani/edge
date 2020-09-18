from edge.dynamics import SlipDynamics, HovershipDynamics
import numpy as np

hov = HovershipDynamics(ground_gravity=0, gravity_gradient=0,
                        control_frequency=10, max_thrust=1, max_altitude=1)
hov.step(np.atleast_1d(0.6), np.atleast_1d(0.0))

slip = SlipDynamics(gravity=0, mass=80.0, stiffness=8200.0, resting_length=1.0,
                    energy=1877)
slip.step(np.atleast_1d(0.9), np.atleast_1d(0.3))