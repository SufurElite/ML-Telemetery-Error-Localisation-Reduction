# The Project's Data

Will need to make a distinction between the June data and the march here and how the util functions were written for each.

## Info On Data
The data is fundamentally straightforward and boils down to the signal strength sent out by a tag to the different reachable Nodes in the network along with the time it was sent. And this is then coupled with the ground truth of the tag's location determined by a GPS.

The former can be found in the BeepData where each row is one description of a signal sent to a Node, and for March the corresponding Ground Truth values are found in the TestInfo and for June in the respective flight csv files.

## March Associated Data Format

The March associated data is less streamlined than the June and may be updated so that they follow the same pattern. But, unlike June's, it involves averaging the values in a given location over 2 minutes. Currently, March's associated data is a JSON with this pattern:

* UniqueTestIdKey (what separates each value in the JSON)
    * The Ground Truth data for the test, i.e. its lat/long GPS, etc.
    * The Data values: a list of JSONs that have all the info 

## June Associated Data Format
June's associatedData JSON consists of two related keys an X and y:

* X: a list of JSONs that have the (2 second) time interval the signals were sent, the tag that sent the signals, and the data JSON for that interval (the nodes it reached and the siganals sent to them)
* y: a list of the actual UTMx, UTMy coordinates at the time interval

There are the same number of X and y values and there exists a 1 to 1 relation in sequential order.