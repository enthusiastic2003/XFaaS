
from node1 import node1 as node1
from node2 import node2 as node2
from node3 import node3 as node3
from node4 import node4 as node4
from node5 import node5 as node5
from node6 import node6 as node6
from node7 import node7 as node7
from finalNode import final as finalNode

from python.src.utils.classes.commons.serwo_objects import SerWOObject, SerWOObjectsList


def function(serwoObject) -> SerWOObject:

    fjyo = node1.function(serwoObject)
    fjyo.set_basepath(serwoObject.get_basepath())
    whbo = node2.function(fjyo)
    whbo.set_basepath(fjyo.get_basepath())
    mydg = node3.function(whbo)
    mydg.set_basepath(whbo.get_basepath())
    mnfq = node4.function(mydg)
    mnfq.set_basepath(mydg.get_basepath())
    umlb = node5.function(mnfq)
    umlb.set_basepath(mnfq.get_basepath())
    wbsm = node6.function(umlb)
    wbsm.set_basepath(umlb.get_basepath())
    ncaa = node7.function(wbsm)
    ncaa.set_basepath(wbsm.get_basepath())
    lonm = finalNode.function(ncaa)
    lonm.set_basepath(ncaa.get_basepath())
    return lonm
