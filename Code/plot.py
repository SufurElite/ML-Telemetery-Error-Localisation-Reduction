from matplotlib import pyplot as plt
from utils import loadData, loadNodes, loadBoundaries, deriveEquation
from multilat import marchPredictions, junePredictions
import os, argparse

# from https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plotOriginal(month="March", threshold=-90):
    print("Plotting " + month + " with " + str(threshold) + " as a threshold")
    redoUtm = False
    if month == "March":
        res = marchPredictions(threshold)
    else:
        res = junePredictions(threshold)
        redoUtm= True
    nodes = loadNodes(redoUtm)
    # removed the boundaries as hardset since
    # the first test is outside the bounds?
    minX,maxX,minY,maxY = loadBoundaries(redoUtm)

    for test in res.keys():


        nodesX = []
        nodesY = []

        for nodeId in nodes.keys():
            nodesX.append(nodes[nodeId]["NodeUTMx"])
            nodesY.append(nodes[nodeId]["NodeUTMy"])

        fig = plt.figure()
        fig.set_size_inches(32, 18)
        ax = fig.add_subplot(111)

        ax.scatter(nodesX, nodesY, c="#0003b8", marker='o', label='Nodes')
        gtX = res[test]["gt"][0]
        gtY = res[test]["gt"][1]

        # if a test is outside the bounds it might be worthwile to log it and remove it
        if (gtX < minX or gtX>maxX) or (gtY < minY or gtY>maxY) :
            print("Outside of the boundaries ",test)

        predX = res[test]["res"][0]
        predY = res[test]["res"][1]

        # plot the actual location
        ax.scatter(gtX,gtY,c='g',marker='o',label='Actual Location')
        # plot the predicted location
        ax.scatter(predX,predY,c='#b800ab',marker='x',label='Predicted Location')
        # draw an arrow between the two with the error labeled
        ax.arrow(predX, predY, gtX-predX, gtY-predY, head_width=0.05, head_length=0.1, color="r", ls=':')
        ax.text((gtX+predX)/2, (gtY+predY)/2, res[test]['error'], fontsize=10, c='#8f0e19',fontweight='heavy')

        # plot the distances to each nodes
        cmap = get_cmap(len(res[test]["nodeDists"]))
        for i in range(len(res[test]["nodeDists"])):
            nodeDist = res[test]["nodeDists"][i]
            nodeLocX = res[test]["nodeLocs"][i][0]
            nodeLocY = res[test]["nodeLocs"][i][1]
            ax.arrow(nodeLocX, nodeLocY, predX-nodeLocX, predY-nodeLocY, head_width=0.05, head_length=0.1, color=cmap(i), ls=':')
            ax.text((nodeLocX+predX)/2, (nodeLocY+predY)/2, nodeDist, fontsize=10, c=cmap(i),fontweight='light', fontstyle='italic')
        plt.legend(loc='best')


        plt.xlabel("UTMx")
        plt.ylabel("UTMy")
        plt.title("The " + month + " data predictions " + str(threshold) + " filter ")

        path = os.getcwd()+"/plots/"+month+"/original model"  + str(threshold)+ " threshold/"
        if not os.path.exists(path):
            os.makedirs(path)

        plt.savefig(path+"/"+str(test)+".png", bbox_inches='tight')
        # bbox_inches removes extra white spaces
        plt.clf()
def plotEquation():
    # 100 linearly spaced numbers
    x = np.linspace(0,200,10000)
    values = utils.deriveEquation()
    print(values)
    y = values[0]*np.exp(x*values[1])+values[2]

    # setting the axes at the centre
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # plot the function
    plt.plot(x,y, 'r')

    # show the plot
    plt.show()

def main(args=None):
    rssiThreshold=-105.16
    if args.rssi!=None:
        rssiThreshold = args.rssi
    if args.month==None:
        print("Please select a month to visualise with --month month, where you can select 'march' or 'june'")
    elif args.month.lower()=="march":
        plotOriginal("March", rssiThreshold)
    elif args.month.lower()=="june":
        plotOriginal("June", rssiThreshold)
    return None

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Plot variables')
    parser.add_argument('--month', dest='month', type=str, help='The month of the data you want to visualise')
    parser.add_argument('--rssi', dest='rssi', type=int, help='rssi filter for the data')

    args = parser.parse_args()
    main(args)
