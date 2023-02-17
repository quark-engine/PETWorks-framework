import json

import PETWorks.federatedlearning as FL
import PETWorks.reidentificationrisk as ReidentificationRisk
import PETWorks.ambiguity as Ambiguity
import PETWorks.precision as Precision
import PETWorks.nonUniformEntropy as NonUniformEntropy
import PETWorks.aecs as AECS
from web.generate import generateWebView

HISTORY = "images/history.png"


def dataProcess(model, gradient, tech, method, **keywordArgs):
    if tech == "FL":
        return FL.dataProcess(model, gradient, tech, method, **keywordArgs)


def PETValidation(recover, origin, tech, **keywordArgs):
    if tech == "FL":
        return FL.PETValidation(recover, origin, tech, **keywordArgs)
    elif tech == "ReidentificationRisk":
        return ReidentificationRisk.PETValidation(recover, origin, tech, **keywordArgs)
    elif tech == "Ambiguity":
        return Ambiguity.PETValidation(recover, origin, tech, **keywordArgs)
    elif tech == "Precision":
        return Precision.PETValidation(recover, origin, tech, **keywordArgs)
    elif tech == "Non-Uniform Entropy":
        return NonUniformEntropy.PETValidation(recover, origin, tech, **keywordArgs)
    elif tech == "AECS":
        return AECS.PETValidation(recover, origin, tech)


def report(result, format):
    if format == "json":
        print(json.dumps(result, indent=4))
        return

    if format == "web":

        originPath = "images/original_image.png"
        recoverPath = "images/recovered_image.png"
        result["origin"].save(originPath)
        result["recover"].save(recoverPath)
        html = generateWebView(
            originPath, recoverPath, HISTORY, result["similarity"]
        )
        with open("output.html", "w") as f:
            f.write(html)
    return