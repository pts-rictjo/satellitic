# Intent
Satellitic is currently under development, but aims to implement methods that aligns with ITU recommendations, with plans to incorporate antenna patterns and functionalities needed for compliance assessment.

## Current state plan
The package aims to further implement:

- full multiple timepoint simulations with trajectory IO control
- antenna pattern functionality
- implementations of interference metrics as per S.1503
- accuracy assessments

# SGP4 and ITU-R S.1503 (orbital prediction engine)
Since S.1503 requires :

- Satellite positions in a geocentric or ECI frame
- Sufficient accuracy for interference and EPFD calculations
- Time-stepped updates of satellite positions

... any validated propagator that meets these requirements should be acceptable. SGP4 is seen as a perfect fit because:

- It computes ECI/TEME positions from TLEs.
- It models Earth’s oblateness, drag, and resonances — enough fidelity for non-GSO regulatory simulations.
- Outputs can be fed directly into S.1503-style EPFD or interference calculations.

So in a compliance workflow, SGP4 provides the orbital prediction engine for satellites, which is exactly what S.1503 needs in its functional description.
