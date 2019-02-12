
from graphics import *
import time
from scipy.interpolate import interp1d
from PIL import Image as NewImage

class GridDisp:

    def __init__(self, shape, title = "std"):
        self.rects = {}
        self.texts = {}
        self.rs = 40
        self.win = GraphWin(title, shape[0]*self.rs,shape[1]*self.rs)
        self.buildGrid(shape)
        self.title = title
        print("Display Setup")

    def buildGrid(self, shape):
        for x in range(shape[0]):
            for y in range(shape[1]):
                self.rects[(x,y)] = Rectangle(Point(x*self.rs, y*self.rs), Point(x*self.rs + self.rs, y*self.rs + self.rs)) 
                R = self.rects[(x,y)]
                R.draw(self.win)
                self.texts[(x,y)] = Text(Point(x*self.rs + 0.4*self.rs, y*self.rs + 0.4 * self.rs), 'NA')
                self.texts[(x,y)].draw(self.win)
                self.texts[(x,y)].setSize(int(self.rs/4))
                self.texts[(x,y)].setTextColor('grey')

    def updateGrid(self, shape, obstacles=[], goals=[], agent=(-1,-1), values=[]):
        Vmin = float('inf')
        Vmax = float('-inf')
        for v in values:
            val = values[v]
            if val<Vmin:
                Vmin = val
            if val>Vmax:
                Vmax = val
        m = interp1d([Vmin,Vmax],[0,254])
        for x in range(shape[0]):
            for y in range(shape[1]):
                R = self.rects[(x,y)]
                if (x,y) in values:
                    R.setFill(color_rgb(255, 0, int(m(values[(x,y)]))))
                else:
                    R.setFill("white")
                for o in obstacles:
                    if x == o[0] and  y == o[1]:
                        R.setFill("black")
                for g in goals:
                    if x == g[0] and  y == g[1]:
                        R.setFill("green")
                if x == agent[0] and  y == agent[1]:
                        R.setFill("blue")
                if values != []:
                    self.texts[(x,y)].setText(str(round(values[x,y],3)))
    def destroy(self):
        self.win.postscript(file=self.title + ".eps", colormode='color')
        # Convert from eps format to gif format using PIL
        #img = NewImage.open("image.eps")
        #img.save(self.title + ".gif", "gif")
        self.win.close()
if __name__ == "__main__":
    G = GridDisp([10,10], [(2,2),(2,3),(3,3)], [(0,0),(9,9)], (5,5), [])
    time.sleep(4)
    G.updateGrid([10,10], [(2,2),(2,3),(3,3)], [(0,0),(9,9)], (5,6), [])
    time.sleep(10)
    G.destroy()