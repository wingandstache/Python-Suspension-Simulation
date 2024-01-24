import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import scipy.sparse.linalg as spla

# Based on this paper: https://www.cs.cmu.edu/~baraff/sigcourse/notesf.pdf

class Simulation:
    class Points:
        def __init__(self):
            self.pointProperties = None
            self.gravityObject = None

        def createPoint(self, position, mass=1, velocity=np.zeros((1,3)), force=np.zeros((1,3)), gravity=True):
            # Make sure arrays are 2d np arrays
            position = np.reshape(position, (1, 3)) # 0-2
            velocity = np.reshape(velocity, (1, 3)) # 3-5
            force = np.reshape(force, (1, 3)) # 6-8
            mass = np.reshape(mass, (1, 1)) # 9
            if self.pointProperties is None:
                self.pointProperties = np.hstack((position, velocity, force, mass))
            else:
                self.pointProperties = np.vstack((self.pointProperties, np.hstack((position, velocity, force, mass))))
            if gravity:
                self.gravityObject.addGravity(self.pointProperties.shape[0] - 1)
            return self.pointProperties.shape[0] - 1 # return index of point
        
        def getDeriviative(self, currentPointState):
            velocity = currentPointState[:, 3:6]
            acceleration = currentPointState[:, 6:9] / np.vstack(currentPointState[:, 9])
            return np.hstack((velocity, acceleration))
        
        def addPosition(self, index, position):
            self.pointProperties[index, 0:3] += position

        def addVelocity(self, index, velocity):
            self.pointProperties[index, 3:6] += velocity

        def addForce(self, index, force):
            self.pointProperties[index, 6:9] += force

        def resetForce(self):
            self.pointProperties[:, 6:9] = 0

        def initGravity(self, gravityObject):
            self.gravityObject = gravityObject

    class Constraints:
        def __init__(self, points, gravityVector=np.array([0, 0, -9.81])):
            self.points = points
            self.springs = self.Springs(self.points)
            self.fixedPoints = self.FixedPoints(self.points)
            self.fixedDistances = self.FixedDistances(self.points)
            self.gravity = self.Gravity(self.points, gravityVector)
            self.points.initGravity(self.gravity)

        def getAllConnectedPointIndexes(self):
            springIndexes = self.springs.getAllConnectedPointIndexes()
            fixedDistanceIndexes = self.fixedDistances.getAllConnectedPointIndexes()
            if springIndexes is None:
                return fixedDistanceIndexes
            elif fixedDistanceIndexes is None:
                return springIndexes
            else:
                indexes = np.vstack((springIndexes, fixedDistanceIndexes))
            return indexes

        class Springs:
            def __init__(self, points):
                self.points = points
                self.springProperties = None

            def createSpring(self, pointIndex1, pointIndex2, springConstant=10, springLength=1, springDamping=1):
                if self.springProperties is None:
                    self.springProperties = np.array([[pointIndex1, pointIndex2, springConstant, springLength, springDamping]])
                else:
                    self.springProperties = np.vstack((self.springProperties, np.array([[pointIndex1, pointIndex2, springConstant, springLength, springDamping]])))
                return self.springProperties.shape[0] - 1 # return index of spring

            def getForces(self, currentPointState):
                forceHolder = np.zeros((self.points.pointProperties.shape[0], 3))
                if self.springProperties is None:
                    return forceHolder
                for i in range(self.springProperties.shape[0]):
                    spring = self.springProperties[i, :]
                    pointIndex1 = spring[0].astype(int)
                    pointIndex2 = spring[1].astype(int)
                    springConstant = spring[2]
                    springNaturalLength = spring[3]
                    springDamping = spring[4]
                    # Calculate force
                    point1Position = currentPointState[pointIndex1, 0:3]
                    point2Position = currentPointState[pointIndex2, 0:3]
                    springVector = point1Position - point2Position
                    springLength = np.linalg.norm(springVector)
                    springDirection = springVector / springLength
                    springForce = -springConstant * (springLength - springNaturalLength) * springDirection
                    # Calculate damping
                    point1Velocity = currentPointState[pointIndex1, 3:6]
                    point2Velocity = currentPointState[pointIndex2, 3:6]
                    relativeVelocity = point1Velocity - point2Velocity
                    dampingForce = -springDamping * np.dot(relativeVelocity, springDirection) * springDirection
                    # Apply forces
                    forceHolder[pointIndex1, :] += springForce + dampingForce
                    forceHolder[pointIndex2, :] += -springForce - dampingForce
                return forceHolder

            def getAllConnectedPointIndexes(self):
                if self.springProperties is None:
                    return None
                return self.points.pointProperties[self.springProperties[:, 0:2].astype(int), 0:3]

        class FixedPoints:
            def __init__(self, points):
                self.points = points
                self.fixedPoints = None

            def createFixedPoint(self, pointIndex):
                pointPosition = self.points.pointProperties[pointIndex, 0:3]
                if self.fixedPoints is None:
                    self.fixedPoints = np.reshape(np.hstack((pointIndex, pointPosition)), (1, 4))
                else:
                    self.fixedPoints = np.vstack((self.fixedPoints, np.hstack((pointIndex, pointPosition))))
                return self.fixedPoints.shape[0] - 1 # return index of fixed point
            
            def behaviorFunction(self, currentPointState):
                # Get the difference between the current position and the fixed position, the array must be 1d vertical and be 3 times the length of the number of fixed points
                differenceArray = self.fixedPoints[:, 1:4] - currentPointState[self.fixedPoints[:, 0].astype(int), 0:3]
                return differenceArray.flatten()
            
            def behaviorFunctionDerivative(self, currentPointState):
                # Get the difference between the velocity of the point and the fixed point
                differenceArray = - currentPointState[self.fixedPoints[:, 0].astype(int), 3:6]
                return differenceArray.flatten()
            
            def getJacobian(self):
                # Get the jacobian of the behavior function
                # Return the pointIndexes and the axes that are constrained
                indexes = np.tile(np.reshape(self.fixedPoints[:, 0].astype(int), (self.fixedPoints.shape[0], 1)), (1, 3)).reshape(-1, 1) # Get 3x the indexes of the fixed points
                axes = np.tile(np.array([0, 1, 2]), (self.fixedPoints.shape[0], 1)).reshape(-1, 1) # Get the axes that are constrained
                indexAxis = np.hstack((indexes, axes)) # Combine the indexes and axes
                # Jacobian is number of constraints by number of fixed points (both multiplied by 3 due to the 3 dimensions)
                return -np.identity(self.fixedPoints.shape[0] * 3), indexAxis
            
            def getJacobianDerivative(self):
                a, b = self.getJacobian()
                return a, b

        class FixedDistances:
            def __init__(self, points):
                self.points = points
                self.fixedDistances = None

            def createFixedDistance(self, pointIndex1, pointIndex2, distance=None):
                if distance is None:
                    distance = np.linalg.norm(self.points.pointProperties[pointIndex1, 0:3] - self.points.pointProperties[pointIndex2, 0:3])
                if self.fixedDistances is None:
                    self.fixedDistances = np.reshape(np.hstack((pointIndex1, pointIndex2, distance)), (1, 3))
                else:
                    self.fixedDistances = np.vstack((self.fixedDistances, np.hstack((pointIndex1, pointIndex2, distance))))
                return self.fixedDistances.shape[0] - 1 # return index of fixed distance

            def behaviorFunction(self, currentPointState):
                # Get the distance between the two points then subtract the distance they are supposed to be apart
                if self.fixedDistances is None:
                    return None
                return np.linalg.norm(currentPointState[self.fixedDistances[:, 0].astype(int), 0:3] - currentPointState[self.fixedDistances[:, 1].astype(int), 0:3], axis=1) - self.fixedDistances[:, 2]
            
            def behaviorFunctionDerivative(self, currentPointState):
                # Get the velocity deriviative
                if self.fixedDistances is None:
                    return None
                # Double check this is working
                numerators = np.sum(2 * np.multiply((currentPointState[self.fixedDistances[:, 0].astype(int), 0:3] - currentPointState[self.fixedDistances[:, 1].astype(int), 0:3]), (currentPointState[self.fixedDistances[:, 0].astype(int), 3:6] - currentPointState[self.fixedDistances[:, 1].astype(int), 3:6])), axis=1)
                denominators = 2 * np.linalg.norm(currentPointState[self.fixedDistances[:, 0].astype(int), 0:3] - currentPointState[self.fixedDistances[:, 1].astype(int), 0:3], axis=1)
                return np.divide(numerators, denominators)

            def getJacobianRow(self, row, currentPointState):
                point1 = currentPointState[self.fixedDistances[row, 0].astype(int), 0:3]
                point2 = currentPointState[self.fixedDistances[row, 1].astype(int), 0:3]
                distance = np.linalg.norm(point1 - point2)
                jacobian = np.array([[(point1[0] - point2[0]), (point1[1] - point2[1]), (point1[2] - point2[2]), (point2[0] - point1[0]), (point2[1] - point1[1]), (point2[2] - point1[2])]]) / distance
                axes = np.array([0, 1, 2])
                indexes = np.array([[self.fixedDistances[row, 0], axes[0]], [self.fixedDistances[row, 0], axes[1]], [self.fixedDistances[row, 0], axes[2]], [self.fixedDistances[row, 1], axes[0]], [self.fixedDistances[row, 1], axes[1]], [self.fixedDistances[row, 1], axes[2]]])
                return jacobian, indexes
            
            def buildGenericJacobian(self, jacobianType, currentPointState):
                jacobian = None
                pointIndexes = None
                if self.fixedDistances is None:
                    return None, None
                for i in range(self.fixedDistances.shape[0]):
                    if jacobian is None:
                        jacobian, pointIndexes = jacobianType(i, currentPointState)
                    else:
                        jacobianAddon = np.zeros(jacobian.shape[1])
                        jacobianToAdd, pointIndexesToAdd = jacobianType(i, currentPointState)
                        for j in range(pointIndexesToAdd.shape[0]):
                            if (pointIndexes == pointIndexesToAdd[j, :]).all(axis=1).any():
                                jacobianAddon[(pointIndexes == pointIndexesToAdd[j, :]).all(axis=1)] = jacobianToAdd[:, j]
                            else:
                                jacobianAddon = np.hstack((jacobianAddon, jacobianToAdd[:, j]))
                                pointIndexes = np.vstack((pointIndexes, pointIndexesToAdd[j, :]))
                        shapeCorrection = np.zeros((jacobian.shape[0], jacobianAddon.shape[0]-jacobian.shape[1]))
                        jacobian = np.hstack((jacobian, shapeCorrection))
                        jacobian = np.vstack((jacobian, jacobianAddon))
                return jacobian, pointIndexes

            def getJacobian(self, currentPointState):
                jacobian, pointIndexes = self.buildGenericJacobian(self.getJacobianRow, currentPointState)
                pass
                return jacobian, pointIndexes
            
            def getJacobianDerivativeRow(self, row, currentPointState):
                point1Position = currentPointState[self.fixedDistances[row, 0].astype(int), 0:3]
                point2Position = currentPointState[self.fixedDistances[row, 1].astype(int), 0:3]
                point1Velocity = currentPointState[self.fixedDistances[row, 0].astype(int), 3:6]
                point2Velocity = currentPointState[self.fixedDistances[row, 1].astype(int), 3:6]
                distance = np.linalg.norm(point1Position - point2Position)
                dervPart = 2 * (point1Position[2] - point2Position[2]) * (point1Velocity[2] - point2Velocity[2]) + 2 * (point1Position[1] - point2Position[1]) * (point1Velocity[1] - point2Velocity[1]) + 2 * (point1Position[0] - point2Position[0]) * (point1Velocity[0] - point2Velocity[0])
                jacobianDeriviative = np.array([[((point1Velocity[0] - point2Velocity[0]) / distance) - (((point1Position[0] - point2Position[0]) * dervPart) / (2 * distance**3)),
                                                 ((point1Velocity[1] - point2Velocity[1]) / distance) - (((point1Position[1] - point2Position[1]) * dervPart) / (2 * distance**3)),
                                                 ((point1Velocity[2] - point2Velocity[2]) / distance) - (((point1Position[2] - point2Position[2]) * dervPart) / (2 * distance**3)),
                                                 ((point2Velocity[0] - point1Velocity[0]) / distance) - (((point2Position[0] - point1Position[0]) * dervPart) / (2 * distance**3)),
                                                 ((point2Velocity[1] - point1Velocity[1]) / distance) - (((point2Position[1] - point1Position[1]) * dervPart) / (2 * distance**3)),
                                                 ((point2Velocity[2] - point1Velocity[2]) / distance) - (((point2Position[2] - point1Position[2]) * dervPart) / (2 * distance**3))]])
                axes = np.array([0, 1, 2])
                indexes = np.array([[self.fixedDistances[row, 0], axes[0]], [self.fixedDistances[row, 0], axes[1]], [self.fixedDistances[row, 0], axes[2]], [self.fixedDistances[row, 1], axes[0]], [self.fixedDistances[row, 1], axes[1]], [self.fixedDistances[row, 1], axes[2]]])
                return jacobianDeriviative, indexes
                                                 
            
            def getJacobianDerivative(self, currentPointState):
                jacobianDeriviative, pointIndexes = self.buildGenericJacobian(self.getJacobianDerivativeRow, currentPointState)
                return jacobianDeriviative, pointIndexes
            
            def getAllConnectedPointIndexes(self):
                if self.fixedDistances is None:
                    return None
                return self.points.pointProperties[self.fixedDistances[:, 0:2].astype(int), 0:3]

        class Gravity:
            def __init__(self, points, gravityVector=np.array([0, 0, -9.81])):
                self.points = points
                self.pointsWithGravity = None
                self.gravityVector = gravityVector

            def addGravity(self, pointIndex):
                if self.pointsWithGravity is None:
                    self.pointsWithGravity = np.array([pointIndex])
                else:
                    self.pointsWithGravity = np.hstack((self.pointsWithGravity, np.array([pointIndex])))
            
            def getForces(self):
                forceHolder = np.zeros((self.points.pointProperties.shape[0], 3))
                forceHolder[self.pointsWithGravity, :] = np.multiply(np.vstack(self.points.pointProperties[self.pointsWithGravity, 9]), self.gravityVector)
                return forceHolder

    def __init__(self, timeStepSize=0.01, gravityVector=np.array([0, 0, -9.81]), viewingBounds=np.array([[-5, 5], [-5, 5], [-5, 5]]), groundLevel=0):
        self.points = self.Points()
        self.constraints = self.Constraints(self.points, gravityVector)
        self.timeStepSize = timeStepSize
        self.viewingBounds = np.array(viewingBounds)
        self.groundLevel = groundLevel
        self.time = 0

        # Set visual properties
        self.bgColor = "#030215"
        self.axisColor = "#353533"
        self.labelColor = "#f2f3d9"
        self.pointColor = "#adadad"
        self.tensionColor = "#ce2929"
        self.compressionColor = "#355fde"
        # Create axes
        xAxis = np.array([[self.viewingBounds[0, 0], 0, 0], [self.viewingBounds[0, 1], 0, 0]])
        yAxis = np.array([[0, self.viewingBounds[1, 0], 0], [0, self.viewingBounds[1, 1], 0]])
        zAxis = np.array([[0, 0, self.viewingBounds[2, 0]], [0, 0, self.viewingBounds[2, 1]]])
        axes = np.array([xAxis, yAxis, zAxis])
        self.axisLines = Line3DCollection(axes, colors=self.axisColor)
        # Axis sublines
        xSubLines = np.array([[[i, viewingBounds[1, 0], 0], [i, viewingBounds[1, 1], 0]] for i in range(viewingBounds[0, 0]+1, viewingBounds[0, 1])])
        ySubLines = np.array([[[viewingBounds[0, 0], i, 0], [viewingBounds[0, 1], i, 0]] for i in range(viewingBounds[1, 0]+1, viewingBounds[1, 1])])
        subLines = np.vstack((xSubLines, ySubLines))
        self.subLineCollection = Line3DCollection(subLines, colors=self.axisColor, linewidths=0.5)

    def createPoint(self, position, mass=1, velocity=np.zeros((1,3)), force=np.zeros((1,3)), gravity=True):
        return self.points.createPoint(position, mass, velocity, force, gravity)
    
    def createSpring(self, point1, point2, springConstant=10, springLength=1, springDamping=1):
        return self.constraints.springs.createSpring(point1, point2, springConstant, springLength, springDamping)
    
    def createFixedPoint(self, point):
        return self.constraints.fixedPoints.createFixedPoint(point)
    
    def createFixedDistance(self, point1, point2, distance=None):
        return self.constraints.fixedDistances.createFixedDistance(point1, point2, distance)
    
    def enableGravity(self, point):
        self.constraints.gravity.addGravity(point)

    def buildJacobian(self, builtJacobian, builtPointIndexes, jacobianToAdd, pointIndexes):
        if (builtJacobian is None and jacobianToAdd is None) or ((builtJacobian.shape[0] == 0) and (jacobianToAdd.shape[0] == 0)): # If both jacobians are empty
            return None, None
        elif builtJacobian is None or (builtJacobian.shape[0] == 0): # If the first jacobian is empty
            builtJacobian = jacobianToAdd
            builtPointIndexes = pointIndexes
        elif jacobianToAdd is None or (jacobianToAdd.shape[0] == 0): # If the second jacobian is empty
            pass
        else:
            jacobianAddon = np.zeros((jacobianToAdd.shape[0], builtJacobian.shape[1])) # Start with a blank matrix the height of the jacobian to add and the width of the built jacobian
            for i in range(pointIndexes.shape[0]):
                if not (builtPointIndexes == pointIndexes[i]).all(axis=1).any(): # If the point index has not been added yet
                    jacobianAddon = np.hstack((jacobianAddon, jacobianToAdd[:, i].reshape(-1, 1))) # Add the column of the jacobian to add to the jacobian addon
                    builtPointIndexes = np.vstack((builtPointIndexes, pointIndexes[i, :])) # Add the point index to the built point indexes
                else:
                    jacobianAddon[:, (builtPointIndexes == pointIndexes[i]).all(axis=1)] = jacobianToAdd[:, i].reshape(-1, 1) # Add the column of the jacobian to add to the jacobian addon
            shapeCorrection = np.zeros((builtJacobian.shape[0], jacobianAddon.shape[1]-builtJacobian.shape[1])) # Create a matrix to correct the shape of the jacobian addon
            builtJacobian = np.hstack((builtJacobian, shapeCorrection)) # Make the jacobian the right shape to add the jacobian addon
            builtJacobian = np.vstack((builtJacobian, jacobianAddon)) # Add the jacobian addon to the jacobian
        return builtJacobian, builtPointIndexes

    def getJacobians(self, currentPointState):
        fixedPointJacobian, fixedPointIndexes = self.constraints.fixedPoints.getJacobian()
        fixedDistanceJacobian, fixedDistanceIndexes = self.constraints.fixedDistances.getJacobian(currentPointState)
        jacobian, pointIndexes = self.buildJacobian(fixedPointJacobian, fixedPointIndexes, fixedDistanceJacobian, fixedDistanceIndexes)
        fixedPointJacobianDeriviative, fixedPointDeriviativeIndexes = self.constraints.fixedPoints.getJacobianDerivative()
        fixedDistanceJacobianDeriviative, fixedDistanceDeriviativeIndexes = self.constraints.fixedDistances.getJacobianDerivative(currentPointState)
        jacobianDeriviative, a = self.buildJacobian(fixedPointJacobianDeriviative, fixedPointDeriviativeIndexes, fixedDistanceJacobianDeriviative, fixedDistanceDeriviativeIndexes)
        pass
        return jacobian, jacobianDeriviative, pointIndexes
    
    def getBehaviorFunction(self, currentPointState):
        fixedBehaviorFunction = self.constraints.fixedPoints.behaviorFunction(currentPointState)
        fixedDistanceBehaviorFunction = self.constraints.fixedDistances.behaviorFunction(currentPointState)
        if fixedBehaviorFunction is None or fixedBehaviorFunction.size == 0:
            return fixedDistanceBehaviorFunction
        elif fixedDistanceBehaviorFunction is None or fixedDistanceBehaviorFunction.size == 0:
            return fixedBehaviorFunction
        else: # Don't need to worry about both being empty because that is covered by the jacobian check
            return np.hstack((fixedBehaviorFunction, fixedDistanceBehaviorFunction))
        
    def getBehaviorFunctionDerivative(self, currentPointState):
        fixedBehaviorFunctionDerivative = self.constraints.fixedPoints.behaviorFunctionDerivative(currentPointState)
        fixedDistanceBehaviorFunctionDerivative = self.constraints.fixedDistances.behaviorFunctionDerivative(currentPointState)
        if fixedBehaviorFunctionDerivative is None or fixedBehaviorFunctionDerivative.size == 0:
            return fixedDistanceBehaviorFunctionDerivative
        elif fixedDistanceBehaviorFunctionDerivative is None or fixedDistanceBehaviorFunctionDerivative.size == 0:
            return fixedBehaviorFunctionDerivative
        else: # Don't need to worry about both being empty because that is covered by the jacobian check
            return np.hstack((fixedBehaviorFunctionDerivative, fixedDistanceBehaviorFunctionDerivative))
        
    def addStabilityDamping(self, dampingConstant):
        velocities = self.points.pointProperties[:, 3:6]
        self.points.pointProperties[:, 6:9] += -dampingConstant * velocities

    def getRigidConstraints(self, currentPointState):
        stabilityConstant = 10
        # Solve for rigid constraints
        # Get the jacobians of the constraints
        jacobian, jacobianDeriviative, pointIndexes = self.getJacobians(currentPointState)
        if jacobian is None: # If there are no constraints
            return
        pointIndexes = pointIndexes.astype(int)
        # Get the velocity of the rigid points
        velocityIndexes = np.copy(pointIndexes)
        velocityIndexes[:, 1] += 3
        velocities = None
        for i in range(velocityIndexes.shape[0]):
            if velocities is None:
                velocities = currentPointState[velocityIndexes[i, 0], velocityIndexes[i, 1]]
            else:
                velocities = np.hstack((velocities, currentPointState[velocityIndexes[i, 0], velocityIndexes[i, 1]]))
        # Get forces on the rigid points
        forceIndexes = np.copy(pointIndexes)
        forceIndexes[:, 1] += 6
        forces = None
        for i in range(forceIndexes.shape[0]):
            if forces is None:
                forces = currentPointState[forceIndexes[i, 0], forceIndexes[i, 1]]
            else:
                forces = np.hstack((forces, currentPointState[forceIndexes[i, 0], forceIndexes[i, 1]]))
        # Get the inverse of the diagonal mass matrix
        invMass = np.diag(1 / currentPointState[pointIndexes[:, 0], 9])
        # Get the energy equation
        behaviorPositionError = self.getBehaviorFunction(currentPointState)
        # Get the time derivative of the energy equation
        behaviorVelocityError = self.getBehaviorFunctionDerivative(currentPointState)
        # Solve for lambda using conjugate gradient algorithm
        A = np.matmul(np.matmul(jacobian, invMass), np.transpose(jacobian))
        b = - np.matmul(jacobianDeriviative, velocities) - np.matmul(np.matmul(jacobian, invMass), forces) - stabilityConstant * behaviorPositionError - stabilityConstant * behaviorVelocityError # Pure magic happens here
        solvedLambda = spla.cg(A, b) # OMG, A THING I DON"T NEED TO MAKE LET"S FUCKIN GOOOOOOOOOOOOOOOOOOOOOO
        correctionForces = np.matmul(np.transpose(jacobian), solvedLambda[0]) # Solve for the correction forces
        forceHolder = np.zeros((currentPointState.shape[0], 3))
        forceHolder[forceIndexes[:, 0], pointIndexes[:, 1]] += correctionForces # Apply the correction forces
        return forceHolder

    def getDeriviative(self, currentPointState):
        forces = self.constraints.springs.getForces(currentPointState)
        forces += self.constraints.gravity.getForces()
        currentPointState[:, 6:9] = forces
        forces += self.getRigidConstraints(currentPointState) # This must be after the non-rigid constraints
        # self.addStabilityDamping(0.1) # Might not be needed
        currentPointState[:, 6:9] = forces
        return self.points.getDeriviative(currentPointState)
        
    def RK4Step(self):
        dt = self.timeStepSize
        self.points.resetForce()
        currentPointState = np.copy(self.points.pointProperties) # Point starting point for following calculations
        k1 = self.getDeriviative(np.copy(currentPointState))
        nextPointState = np.copy(currentPointState)
        nextPointState[:, 0:6] += k1 * (dt / 2)
        k2 = self.getDeriviative(nextPointState)
        nextPointState = np.copy(currentPointState)
        nextPointState[:, 0:6] += k2 * (dt / 2)
        k3 = self.getDeriviative(nextPointState)
        nextPointState = np.copy(currentPointState)
        nextPointState[:, 0:6] += k3 * dt
        k4 = self.getDeriviative(nextPointState)
        deriviative = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.points.pointProperties[:, 0:6] += deriviative * dt
        self.time += dt
        pass

def initVisualize(simulation):
    fig = plt.figure(facecolor=simulation.bgColor)
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim3d(simulation.viewingBounds[0, 0], simulation.viewingBounds[0, 1])
    ax.set_ylim3d(simulation.viewingBounds[1, 0], simulation.viewingBounds[1, 1])
    ax.set_zlim3d(simulation.viewingBounds[2, 0], simulation.viewingBounds[2, 1])
    ax.add_collection(simulation.axisLines)
    ax.add_collection(simulation.subLineCollection)
    ax.scatter(simulation.points.pointProperties[:, 0], simulation.points.pointProperties[:, 1], simulation.points.pointProperties[:, 2], c=simulation.pointColor) # plot points

    ax.set_facecolor(simulation.bgColor)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    rcParams['axes3d.grid'] = False
    rcParams['xtick.color'] = (1.0, 1.0, 1.0, 0.0)
    
    lines = Line3DCollection(simulation.constraints.getAllConnectedPointIndexes(), colors=simulation.pointColor)
    ax.add_collection(lines)
    ax.annotate("Time: " + str(round(simulation.time, 2)), xy=(0.01, 0.95), xycoords="axes fraction", color=simulation.labelColor)
    return fig

def simulationStep(i, simulation, fig):
    plt.cla()
    simulation.RK4Step()
    ax = fig.axes[0]
    ax.set_xlim3d(simulation.viewingBounds[0, 0], simulation.viewingBounds[0, 1])
    ax.set_ylim3d(simulation.viewingBounds[1, 0], simulation.viewingBounds[1, 1])
    ax.set_zlim3d(simulation.viewingBounds[2, 0], simulation.viewingBounds[2, 1])
    ax.add_collection(simulation.axisLines)
    ax.add_collection(simulation.subLineCollection)
    ax.scatter(simulation.points.pointProperties[:, 0], simulation.points.pointProperties[:, 1], simulation.points.pointProperties[:, 2], c=simulation.pointColor) # plot points
    lines = Line3DCollection(simulation.constraints.getAllConnectedPointIndexes(), colors=simulation.pointColor)
    ax.add_collection(lines)
    ax.annotate("Time: " + str(round(simulation.time, 2)), xy=(0.01, 0.95), xycoords="axes fraction", color=simulation.labelColor)
    pass

def runSimulation(simulation):
    fig = initVisualize(simulation)
    ani = anim.FuncAnimation(fig, simulationStep, fargs=(simulation, fig), interval=1)
    plt.show()

if __name__ == "__main__":
    sim = Simulation(timeStepSize=0.05, viewingBounds=np.array([[-5, 5], [-5, 5], [0, 10]]))
    # Suspension
    p1 = sim.createPoint([[-1], [-1], [1.9]]) # Lower A Arm
    p2 = sim.createPoint([[-1], [1], [2.1]])
    p3 = sim.createPoint([[2], [-0.21], [2]])
    sim.createFixedDistance(p1, p3)
    sim.createFixedDistance(p2, p3)
    sim.createFixedPoint(p1)
    sim.createFixedPoint(p2)
    p4 = sim.createPoint([[-0.5], [-1], [5.1]]) # Upper A Arm
    p5 = sim.createPoint([[-0.5], [1], [4.9]])
    p6 = sim.createPoint([[1.6], [0.2], [5]])
    sim.createFixedDistance(p4, p6)
    sim.createFixedDistance(p5, p6)
    sim.createFixedPoint(p4)
    sim.createFixedPoint(p5)
    sim.createFixedDistance(p3, p6)
    p7 = sim.createPoint([[-0.5], [0], [5]]) # Spring mounts
    p8 = sim.createPoint([[0.5], [0], [2.01]])
    sim.createFixedDistance(p1, p8)
    sim.createFixedDistance(p2, p8)
    sim.createFixedDistance(p3, p8)
    sim.createFixedPoint(p7)
    sim.createSpring(p7, p8, springConstant=90, springLength=2, springDamping=1)
    p9 = sim.createPoint([[2], [0.8], [3]]) # Upright
    p10 = sim.createPoint([[2.5], [0], [3.5]])
    sim.createFixedDistance(p9, p10)
    sim.createFixedDistance(p9, p3)
    sim.createFixedDistance(p9, p6)
    sim.createFixedDistance(p10, p3)
    sim.createFixedDistance(p10, p6)
    p11 = sim.createPoint([[-1], [1], [3.5]]) # Steering arm mount
    sim.createFixedDistance(p11, p9)
    sim.createFixedPoint(p11)
    # # Test case
    # p1 = sim.createPoint([[0], [0], [4]])
    # p2 = sim.createPoint([[3], [3], [4]])
    # sim.createFixedDistance(p1, p2)
    # sim.createFixedPoint(p1)
    runSimulation(sim)