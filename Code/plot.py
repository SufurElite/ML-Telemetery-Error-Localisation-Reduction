from matplotlib import pyplot as plt
from utils import loadData, loadNodes, loadBoundaries, deriveEquation, loadSections, pointToSection
from multilat import marchPredictions, junePredictions
import os, argparse, utm

# from https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plotNodes(rewriteUTM=True, plotSections=True, isAllData = False):
    """ Function visualise the node locations and their relative distance to one another """
    # if we are going to save the node setup, this should be the relative file path
    filePath = "/plots/Nodes/NodeSetup.png"
    # load in all the related values
    grid, sections, nodes = loadSections()

    # plot variables
    fig = plt.figure()
    fig.set_size_inches(32, 18)
    ax = fig.add_subplot(111)
    
    # for each node key we want to have a different color
    nodeKeys = list(nodes.keys())
    cmap = get_cmap(len(nodeKeys))
    # plot each node as a different color, with its id 10 below
    for idx in range(len(nodeKeys)):
        curX = nodes[nodeKeys[idx]]["NodeUTMx"]
        curY = nodes[nodeKeys[idx]]["NodeUTMy"]
        ax.text(curX, curY-10, nodeKeys[idx], fontsize=10,fontweight='heavy')
        ax.scatter(curX, curY, c=cmap(idx), marker='o')

    # Now go through the grid and plot the relative distances between each node of form (i,j)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            # check if there is a value for dist to the right
            if j!=len(grid[j])-1 and grid[i][j][1]!=0:
                ax.arrow(nodes[grid[i][j][0]]["NodeUTMx"], nodes[grid[i][j][0]]["NodeUTMy"], nodes[grid[i][j+1][0]]["NodeUTMx"]-nodes[grid[i][j][0]]["NodeUTMx"], nodes[grid[i][j+1][0]]["NodeUTMy"]-nodes[grid[i][j][0]]["NodeUTMy"], head_width=0.05, head_length=0.1, color="r", ls=':')       
                ax.text((nodes[grid[i][j+1][0]]["NodeUTMx"]+nodes[grid[i][j][0]]["NodeUTMx"])/2, (nodes[grid[i][j+1][0]]["NodeUTMy"]+nodes[grid[i][j][0]]["NodeUTMy"])/2, grid[i][j][1], fontsize=10, c='b',fontweight='heavy')
            # check if there is a value for the dist to the top
            if i!=0 and grid[i][j][2]!=0:
                ax.arrow(nodes[grid[i][j][0]]["NodeUTMx"], nodes[grid[i][j][0]]["NodeUTMy"], nodes[grid[i-1][j][0]]["NodeUTMx"]-nodes[grid[i][j][0]]["NodeUTMx"], nodes[grid[i-1][j][0]]["NodeUTMy"]-nodes[grid[i][j][0]]["NodeUTMy"], head_width=0.05, head_length=0.1, color="r", ls=':')       
                ax.text((nodes[grid[i-1][j][0]]["NodeUTMx"]+nodes[grid[i][j][0]]["NodeUTMx"])/2, (nodes[grid[i-1][j][0]]["NodeUTMy"]+nodes[grid[i][j][0]]["NodeUTMy"])/2, grid[i][j][2], fontsize=10, c='b',fontweight='heavy')
    
    # if we're plotting the sections go through each and put the index in its center
    if plotSections:
        for i in range(len(sections)):
            ax.text((sections[i][0][0]+sections[i][1][0])/2, (sections[i][0][1]+sections[i][1][1])/2, i, fontsize=16, c='g',fontweight='heavy')
        # update the filepath accordingly to show for sections
        filePath = "/plots/Nodes/NodeSetupSections.png"
    # add the axis labels and the title
    plt.xlabel("UTMx")
    plt.ylabel("UTMy")
    plt.title("The setup of the Node Grid")
    if not isAllData:
        plt.savefig(os.getcwd()+filePath)
    # if we're not just saving the file return the fig, ax, sections for further plotting
    return fig, ax, sections

def plotGridWithPoints(data, isSections=True):
    """ This takes in a list of data points to be plotted on the grid 
        and whether or not you want to see the sections in the grid too
        and plots accordingly """
    # Get the plot fig, ax and sections from plotNodes as a base
    fig, ax, sections = plotNodes(True, isSections, True)
    # then plot all the data points on top of it
    for point in data:
        ax.scatter(point[0],point[1])
    # show the result
    plt.show()

def plotAllData(month="June", isSections=True, combined=False, onlyOutside = True):
    # load in a fig and ax with the node grid already displayed inplace
    fig, ax, sections = plotNodes(True, isSections, True)
    # set the file path based on month and whether we're including sections and whether 
    # only we're showing values outside the grid
    filePath = "/plots/"+month+"/Data"
    if isSections:
        filePath+="WithSections"
    if onlyOutside:
        filePath+="OnlyOutside"
    filePath+=".png"
    # load in all the data lat longs
    latLongs = loadData(month)["y"]
    # plot each data point in lat longs
    for latLong in latLongs:
        # Ignore any lat long that falls into this area
        if(latLong == [0,0] or latLong == [0.754225181062586, -12.295563892977972] or latLong == [4.22356791831445, -68.85519277364834]): continue
        # find the utm of this data point
        utmVals = utm.from_latlon(latLong[0], latLong[1])
        # if we only want to plot the outside values, check for the values in -1 sections
        if not onlyOutside:
            ax.scatter(utmVals[0],utmVals[1])
        elif pointToSection(utmVals[0],utmVals[1],sections)==-1:
            ax.scatter(utmVals[0],utmVals[1])
    plt.savefig(os.getcwd()+filePath)

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
    values = deriveEquation()
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
    if args.eq.lower()=='y':
        plotEquation()
    elif args.allData.lower()=='y':
        plotAllData()
    elif args.month==None:
        print("Please select a month to visualise with --month month, where you can select 'march' or 'june'")
    elif args.month.lower()=="march":
        plotOriginal("March", rssiThreshold)
    elif args.month.lower()=="june":
        plotOriginal("June", rssiThreshold)
    return None

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Plot variables')
    parser.add_argument('--equation', dest='eq', type=str,help="Do you want to plot the equation? y/n")
    parser.add_argument('--allData', dest='eq', type=str,help="Do you want to plot all the data? y/n")
    parser.add_argument('--month', dest='month', type=str, help='The month of the data you want to visualise')
    parser.add_argument('--rssi', dest='rssi', type=int, help='rssi filter for the data')

    args = parser.parse_args()
    main(args)
