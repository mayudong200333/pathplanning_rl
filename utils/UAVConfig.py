import time
import vtk
import copy
import numpy as np

class UAVConfig:
    def __init__(self):
        self.basic_directions = [i*np.pi/16 for i in range(-8,10,2)]
        self.original_observation_length = 15

        self.level = 100.0
        self.max_speed = 50.0
        self.min_distance_to_target = 10.0
        self.real_action_range = np.array([0.25, 2.0])
        self.min_distance_to_obstacle = 1.0
        self.min_initial_starts = 200.0
        self.expand = 64
        self.num_circle = 15
        self.radius = 60

        self.lowest = 30
        self.period = 180
        self.delta = 23
        self.total = 10

        # range finder parameters
        self.scope = 100.0
        self.min_step = 0.1

        # rendering parameters
        self.margin = 1
        self.camera_alpha = 0.2

        assert self.min_distance_to_obstacle > 0
        assert self.period > self.scope - self.radius

class ToolBox:
    def __init__(self):
        pass

    def SetCamera(self, camera=None, position=(1000,1000,500), direction=0):
        if camera:
            position[0] = position[0] - np.cos(direction / 360 * 2 * np.pi)*40
            position[1] = position[1] - np.sin(direction / 360 * 2 * np.pi)*40
            position[2] = position[2]
            camera.SetPosition(position)

            position[0] = position[0] + np.cos(direction / 360 * 2 * np.pi) * 40
            position[1] = position[1] + np.sin(direction / 360 * 2 * np.pi) * 40
            position[2] = position[2]
            camera.SetFocalPoint(position)

            camera.Elevation(15)
            camera.Azimuth(0)
        else:
            camera = vtk.vtkCamera()
            camera.SetViewUp(0, 0, 1)
            camera.Zoom(0.8)
            camera.Elevation(200)
            return camera

    def CreateGround(self, size=4000):
        # create plane source
        plane = vtk.vtkPlaneSource()
        plane.SetXResolution(100)
        plane.SetYResolution(100)
        plane.SetCenter(0.3, 0.3, 0)
        plane.SetNormal(0, 0, 1)

        # mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())

        # actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetColor(211/255,211/255,211/255)
        transform = vtk.vtkTransform()
        transform.Scale(size, size, 1)
        actor.SetUserTransform(transform)

        return actor

    def CreateCoordinates(self, size=1000):
        # create coordinate axes in the render window
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(300, 300, 300)  # Set the total length of the axes in 3 dimensions
        axes.SetShaftType(0)

        axes.SetCylinderRadius(0.02)
        axes.SetSphereRadius(1)
        axes.GetXAxisCaptionActor2D().SetWidth(0.01)
        axes.GetYAxisCaptionActor2D().SetWidth(0.01)
        axes.GetZAxisCaptionActor2D().SetWidth(0.01)
        return axes

    def CreateCylinder(self, p1, p2, r=30, color=(1.0,1.0,1.0)):
        x, y, h = p1[0], p1[1], p2[2]
        line = vtk.vtkLineSource()
        line.SetPoint1(x, y, 0)
        line.SetPoint2(x, y, h)

        tubefilter = vtk.vtkTubeFilter()
        tubefilter.SetInputConnection(line.GetOutputPort())
        tubefilter.SetRadius(r)
        tubefilter.SetNumberOfSides(30)
        tubefilter.CappingOn()

        cylinderMapper = vtk.vtkPolyDataMapper()
        cylinderMapper.SetInputConnection(tubefilter.GetOutputPort())
        cylinderActor = vtk.vtkActor()
        cylinderActor.GetProperty().SetColor(color)
        cylinderActor.SetMapper(cylinderMapper)
        return cylinderActor

    def CreateLine(self, p1, p2, color=(56/255,94/255,16/255)):
        line = vtk.vtkLineSource()
        line.SetPoint1(p1)
        line.SetPoint2(p2)
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputConnection(line.GetOutputPort())
        lineActor = vtk.vtkActor()
        lineActor.GetProperty().SetColor(color)
        lineActor.GetProperty().SetLineWidth(2.0)
        lineActor.SetMapper(lineMapper)

        return lineActor

    def CreateArrow(self, angle=90, scale=(5, 5, 5), position=(100, 100, 100),
                    color=(255 / 255, 0 / 255, 0 / 255)):
        pointer = vtk.vtkArrowSource()
        pointer.SetTipLength(0.15)
        pointer.SetTipRadius(0.08)
        pointer.SetTipResolution(100)
        pointer.SetShaftRadius(0.015)
        pointer.SetShaftResolution(100)

        transform = vtk.vtkTransform()
        transform.RotateWXYZ(angle, 0, 0, 1)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputConnection(pointer.GetOutputPort())
        transformFilter.Update()

        pointerActor = vtk.vtkActor()
        pointerActor.SetScale(scale)
        pointerActor.AddPosition(position)
        pointerActor.GetProperty().SetColor(color)

        pointerMapper = vtk.vtkPolyDataMapper()
        pointerMapper.SetInputConnection(transformFilter.GetOutputPort())

        pointerActor.SetMapper(pointerMapper)

        return pointerActor

    def CreateSphere(self, p, r, color=(199/255,97/255,20/255), opacity=1.0):
        ball = vtk.vtkSphereSource()
        ball.SetRadius(r)
        ball.SetCenter(p[0], p[1], p[2])
        ball.SetPhiResolution(16)
        ball.SetThetaResolution(32)

        ballMapper = vtk.vtkPolyDataMapper()
        ballMapper.SetInputConnection(ball.GetOutputPort())
        ballActor = vtk.vtkActor()
        ballActor.GetProperty().SetColor(color)
        ballActor.SetMapper(ballMapper)
        ballActor.GetProperty().SetOpacity(opacity)

        return ballActor








if __name__ == '__main__':
    a = UAVConfig()
    print(a.directions)