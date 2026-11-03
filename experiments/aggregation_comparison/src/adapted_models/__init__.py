"""Independent patient-bag adaptations; historical baseline modules stay unchanged."""
from .dgr_mil import DGRMILPatient
from .acmil import ACMILPatient

def build(method, d=1280):
    return {'dgr_mil':DGRMILPatient,'acmil':ACMILPatient}[method](d)
