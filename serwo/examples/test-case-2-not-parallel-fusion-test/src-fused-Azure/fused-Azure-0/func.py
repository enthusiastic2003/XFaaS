
from node1 import node1 as TaskA
from node2 import node2 as TaskB
from node3 import node3 as TaskC
from finalNode import final as FinalTask

from python.src.utils.classes.commons.serwo_objects import SerWOObject, SerWOObjectsList


def function(serwoObject) -> SerWOObject:

    jsii = TaskA.function(serwoObject)
    jsii.set_basepath(serwoObject.get_basepath())
    qjen = TaskB.function(jsii)
    qjen.set_basepath(jsii.get_basepath())
    tqqy = TaskC.function(qjen)
    tqqy.set_basepath(qjen.get_basepath())
    fjjh = FinalTask.function(tqqy)
    fjjh.set_basepath(tqqy.get_basepath())
    return fjjh
