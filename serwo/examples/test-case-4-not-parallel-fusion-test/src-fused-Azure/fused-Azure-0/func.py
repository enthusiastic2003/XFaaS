
from node1 import node1 as node1
from node2 import node2 as node2
from node3 import node3 as node3
from node4 import node4 as node4
from node5 import node5 as node5
from finalNode import final as finalNode

from python.src.utils.classes.commons.serwo_objects import SerWOObject, SerWOObjectsList


def function(serwoObject) -> SerWOObject:

    wgtb = node1.function(serwoObject)
    wgtb.set_basepath(serwoObject.get_basepath())
    iirb = node2.function(wgtb)
    iirb.set_basepath(wgtb.get_basepath())
    unvk = node3.function(iirb)
    unvk.set_basepath(iirb.get_basepath())
    xrut = node4.function(unvk)
    xrut.set_basepath(unvk.get_basepath())
    kmle = node5.function(xrut)
    kmle.set_basepath(xrut.get_basepath())
    ukpf = finalNode.function(kmle)
    ukpf.set_basepath(kmle.get_basepath())
    return ukpf
