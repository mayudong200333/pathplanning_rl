import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as pat

matplotlib.use('Qt5Agg')

class Visualize:
    def __init__(self,map_matrix,history,agent_num=1,animation=False):
        self.map_matrix = map_matrix
        self.history = history
        self.agent_num = agent_num
        self.animation = animation

        plt.figure()
        self.draw_graph()
        self.draw_path()
        plt.show()

    def draw_graph(self):
        lenrow = len(self.map_matrix)
        lencol = len(self.map_matrix[0])
        plt.xlim(-1,lenrow+1)
        plt.ylim(-1,lencol+1)
        ax = plt.subplot()
        for i in range(lenrow):
            for j in range(lencol):
                if self.map_matrix[i][j] == 0:
                    plt.scatter(i,j,s=10,color="lightblue")
                else:
                    xc = i - 1/2
                    yc = j - 1/2
                    rec = pat.Rectangle(xy=(xc,yc),width=1,height=1,angle=0,color="black")
                    ax.add_patch(rec)

    def draw_path(self):
        colorlis = [(random.random(),random.random(),random.random()) for _ in range(self.agent_num)]
        for i in range(self.agent_num):
            plt.scatter(self.history[i]['state'][0][0],self.history[i]['state'][0][1],marker='D',color=colorlis[i],label='agent'+str(i))
            plt.scatter(self.history[i]['goal'][0][0],self.history[i]['goal'][0][1],marker='*',color=colorlis[i])

        plt.legend(bbox_to_anchor=(1.05,1),loc='upper left',borderaxespad = 0,fontsize = 10)

        for i in range(len(self.history[0]['state'])):
            for j in range(self.agent_num):
                state = self.history[j]['state']
                try:
                    plt.plot([state[i][0],state[i+1][0]],[state[i][1],state[i+1][1]],color=colorlis[j])
                except:
                    pass

            if self.animation:
                plt.pause(0.05)