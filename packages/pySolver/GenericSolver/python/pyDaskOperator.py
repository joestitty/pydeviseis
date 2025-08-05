# Module containing the definition of Dask-based operator class
from pyDaskVector import DaskVector
from pyDaskVector import scatter_large_data
from pyVector import vector
import pyOperator as Op
import dask.distributed as daskD
from dask_util import DaskClient
import numpy as np

# For checking if a single argument was passed to a constructor
from collections.abc import Iterable
import time  # DEBUG


def call_constructor(constr, args, kwargs=None):
    """Function to call the constructor"""
    if kwargs is None:
        if isinstance(args, Iterable):
            op = constr(*args)
        else:
            op = constr(args)
    else:
        if isinstance(args, Iterable):
            op = constr(*args, **kwargs)
        else:
            op = constr(args, **kwargs)
    return op


def call_getDomain(opObj):
    """Function to call getDomain method"""
    domain = opObj.getDomain()
    return domain


def call_getRange(opObj):
    """Function to call getRange method"""
    range = opObj.getRange()
    return range


def call_forward(opObj, add, model, data):
    """Function to call forward operator"""
    res = opObj.forward(add, model, data)
    return res


def call_adjoint(opObj, add, model, data):
    """Function to call adjoint operator"""
    res = opObj.adjoint(add, model, data)
    return res


def getNdfuture(vecObj):
    """Function to obtain NdArray as a future object"""
    Nd = vecObj.getNdArray()
    return Nd


def call_func_name(opObj, func_name, *args):
    """Function to call a method by name"""
    fun2call = getattr(opObj, func_name)
    res = fun2call(*args)
    return res


def add_from_NdArray(vecObj, NdArray):
    """Function to add vector values from numpy array"""
    vecObj.getNdArray()[:] += NdArray
    return


class DaskOperator(Op.Operator):
    """
       Class to apply multiple operators in parallel through Dask and DaskVectors
    """

    def __init__(self, dask_client, op_constructor, op_args, chunks, **kwargs):
        """
           Dask Operator constructor
           dask_client = [no default] - DaskClient; client object to use when submitting tasks (see dask_util module)
           op_constructor = [no default] - pointer to function; Pointer to constructor
           op_args = [no default] - list; List containing lists of arguments to run the constructor. It can instantiate the same operator on multiple workers or different ones if requested by passing a list of list of arguments (e.g., [(arg1,arg2,arg3,...)])
           chunks = [no default] - list; List defininig how many operators wants to instantiated. Note, the list must contain the same number of elements as the number of Dask workers present in the DaskClient.
           setbackground_func_name = [None] - string; Name of the function to set the model point on which the Jacobian is computed. See NonLinearOperator in pyOperator module.
           spread_op = [None] - DaskSpreadOp; Spreading operator to distribute a model vector to the set_background functions
           set_aux_name = [None] - string; Name of the function to set the auxiliary vector. Useful for VpOperator.
           spread_op_aux = [None] - DaskSpreadOp; Spreading operator to distribute a auxiliary vector to the set_aux functions
        """
        # Client to submit tasks
        if not isinstance(dask_client, DaskClient):
            raise TypeError("Passed client is not a Dask Client object!")
        if not isinstance(op_args, list):
            raise TypeError("Passed operator arguments not a list!")
        self.dask_client = dask_client
        self.client = self.dask_client.getClient()
        wrkIds = self.dask_client.getWorkerIds()
        N_wrk = self.dask_client.getNworkers()
        # Check if number of provided chunks is the same as workers
        if (len(chunks) != N_wrk):
            raise ValueError(
                "Number of provide chunks (%s) different than the number of workers (%s)" % (len(chunks), N_wrk))
        # Check if many arguments are passed to construct different operators
        N_args = len(op_args)
        N_ops = int(np.sum(chunks))
        if N_args > 1:
            if N_args != N_ops:
                raise ValueError(
                    "Number of lists of arguments (%s) different than the number of requested operators (%s)" % (
                        N_args, N_ops))
        else:
            if N_ops > 1:
                op_args = [op_args for ii in range(N_ops)]

        # Check if kwargs for constructor was provided:
        op_kwargs = self.set_background_name = kwargs.get("op_kwargs", None)
        if op_kwargs is not None:
            N_kwargs = len(op_kwargs)
            if N_kwargs != N_args:
                raise ValueError("Length of kwargs (%d) different than args (%d)!"%(N_kwargs, N_args))
        else:
            op_kwargs = [None]*N_args

        # Instantiation of the operators on each worker
        self.dask_ops = []
        for iwrk, wrkId in enumerate(wrkIds):
            for iop in range(chunks[iwrk]):
                self.dask_ops.append(
                    self.client.submit(call_constructor, op_constructor, op_args.pop(0), op_kwargs.pop(0), workers=[wrkId], pure=False))
        daskD.wait(self.dask_ops)
        for idx in range(len(self.dask_ops)):
            if (self.dask_ops[idx].status == 'error'):
                print("Error for dask operator %s"%(idx))
                print(self.dask_ops[idx].result())
        # Creating domain and range of the Dask operator
        dom_vecs = []  # List of remote domain vectors
        rng_vecs = []  # List of remote range vectors
        for op in self.dask_ops:
            dom_vecs.append(self.client.submit(call_getDomain, op, pure=False))
            rng_vecs.append(self.client.submit(call_getRange, op, pure=False))
        daskD.wait(dom_vecs + rng_vecs)
        self.domain = DaskVector(self.dask_client, dask_vectors=dom_vecs)
        self.range = DaskVector(self.dask_client, dask_vectors=rng_vecs)
        # Set background function name "necessary for non-linear operator Jacobian"
        self.set_background_name = kwargs.get("setbackground_func_name", None)
        if self.set_background_name:
            # Creating a spreading operator useful
            self.Sprd = kwargs.get("spread_op", None)
            if self.Sprd:
                if not isinstance(self.Sprd, DaskSpreadOp):
                    raise TypeError("Provided spread_op not a DaskSpreadOp class!")
                self.model_tmp = self.Sprd.getRange().clone()
        # Set aux function name "necessary for VP operator"
        self.set_aux_name = kwargs.get("set_aux_name", None)
        if self.set_aux_name:
            # Creating a spreading operator useful
            self.SprdAux = kwargs.get("spread_op_aux", None)
            if self.SprdAux:
                if not isinstance(self.SprdAux, DaskSpreadOp):
                    raise TypeError("Provided spread_op_aux not a DaskSpreadOp class!")
                self.tmp_aux = self.SprdAux.getRange().clone()
        return

    def __str__(self):
        return " DaskOp"

    def forward(self, add, model, data):
        """Forward Dask operator"""
        if not isinstance(model, DaskVector):
            raise TypeError("Model vector must be a DaskVector!")
        if not isinstance(data, DaskVector):
            raise TypeError("Data vector must be a DaskVector!")
        # Dimensionality check
        self.checkDomainRange(model, data)
        add = [add] * len(self.dask_ops)
        fwd_ftr = self.client.map(call_forward, self.dask_ops, add, model.vecDask, data.vecDask, pure=False)
        daskD.wait(fwd_ftr)
        return

    def adjoint(self, add, model, data):
        """Adjoint Dask operator"""
        if not isinstance(model, DaskVector):
            raise TypeError("Model vector must be a DaskVector!")
        if not isinstance(data, DaskVector):
            raise TypeError("Data vector must be a DaskVector!")
        # Dimensionality check
        self.checkDomainRange(model, data)
        add = [add] * len(self.dask_ops)
        adj_ftr = self.client.map(call_adjoint, self.dask_ops, add, model.vecDask, data.vecDask, pure=False)
        daskD.wait(adj_ftr)
        return

    def set_background(self, model):
        """Function to call set_background function of each dask operator"""
        if self.set_background_name == None:
            raise NameError("setbackground_func_name was not defined when constructing the operator!")
        if (self.Sprd):
            self.Sprd.forward(False, model, self.model_tmp)
            model = self.model_tmp
        setbkg_ftr = self.client.map(call_func_name, self.dask_ops,
                                     [self.set_background_name] * self.dask_client.getNworkers(), model.vecDask,
                                     pure=False)
        daskD.wait(setbkg_ftr)
        return

    def set_aux(self, aux_vec):
        """Function to call set_nl or set_lin_jac functions of each dask operator"""
        if self.set_aux_name == None:
            raise NameError("set_aux_name was not defined when constructing the operator!")
        if (self.SprdAux):
            self.SprdAux.forward(False, aux_vec, self.tmp_aux)
            aux_vec = self.tmp_aux
        setaux_ftr = self.client.map(call_func_name, self.dask_ops,
                                     [self.set_aux_name] * self.dask_client.getNworkers(), aux_vec.vecDask, pure=False)
        daskD.wait(setaux_ftr)
        return


class DaskSpreadOp(Op.Operator):
    """
       Class to spread/stack single vector to/from multiple copies on different workers:
            | v1 |   | I |
       fwd: | v2 | = | I | v  adj: | v | = | I | v1 + | I | v2 + | I | v3
            | v3 |   | I |
    """

    def __init__(self, dask_client, domain, chunks):
        """
           Dask Operator constructor
           dask_client = [no default] - DaskClient; client object to use when submitting tasks (see dask_util module)
           domain   = [no default] - vector class; Vector template to be spread/stack (note this is also the domain of the operator)
           chunks      = [no default] - list; List defininig how many operators wants to instantiated. Note, the list must contain the same number of elements as the number of Dask workers present in the DaskClient.
        """
        if not isinstance(dask_client, DaskClient):
            raise TypeError("Passed client is not a Dask Client object!")
        if not isinstance(domain, vector):
            raise TypeError("domain is not a vector-derived object!")
        self.dask_client = dask_client
        self.client = self.dask_client.getClient()
        self.chunks = chunks
        self.setDomainRange(domain, DaskVector(self.dask_client, vector_template=domain, chunks=chunks))
        return

    def __str__(self):
        return " DaskSpr"

    def forward(self, add, model, data):
        """Forward operator"""
        if not isinstance(data, DaskVector):
            raise TypeError("Data vector must be a DaskVector!")
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        # Model vector checking
        if isinstance(model, DaskVector):
            # Getting the future to the first vector in the Dask vector
            modelNd = self.client.submit(getNdfuture, model.vecDask[0], pure=False)
        else:
            # Getting the numpy array to the local model vector
            modelNd = model.getNdArray()

        # Spreading model array to workers
        if len(self.chunks) == self.dask_client.getNworkers():
            dataVecList = data.vecDask.copy()
            for iwrk, wrkId in enumerate(self.dask_client.getWorkerIds()):
                # arrD = self.client.scatter(modelNd, workers=[wrkId])
                # daskD.wait(arrD)
                arrD = scatter_large_data(modelNd, wrkId, self.client)
                for ii in range(self.chunks[iwrk]):
                    daskD.wait(
                        self.client.submit(add_from_NdArray, dataVecList.pop(0), arrD, workers=[wrkId], pure=False))
        else:
            # Letting Dask handling the scattering of the data (not ideal)
            futures = self.client.map(add_from_NdArray, data.vecDask, [modelNd] * len(data.vecDask), pure=False)
            daskD.wait(futures)
        return

    def adjoint(self, add, model, data):
        """Adjoint operator"""
        if not isinstance(data, DaskVector):
            raise TypeError("Data vector must be a DaskVector!")
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        if isinstance(model, DaskVector):
            arrD = self.client.map(getNdfuture, data.vecDask, pure=False)
            daskD.wait(arrD)
            sum_array = self.client.submit(np.sum, arrD, axis=0, pure=False)
            daskD.wait(sum_array)
            # Getting the future to the first vector in the Dask vector
            daskD.wait(self.client.submit(add_from_NdArray, model, sum_array, pure=False))
        else:
            arrD_list = data.getNdArray()
            # Getting the numpy array to the local model vector
            modelNd = model.getNdArray()
            for arr_i in arrD_list:
                modelNd[:] += arr_i
        return


class DaskCollectOp(Op.Operator):
    """
       Class to Collect/Scatter a Dask vector into/from a local vector
    """

    def __init__(self, domain, range):
        """
           Dask Collect constructor
           :param domain : - DaskVector; Dask vector to be collected from remote
           :param range : - Vector;  Vector class to be locally stored
        """
        if not isinstance(domain, DaskVector):
            raise TypeError("domain is not a DaskVector object!")
        if isinstance(range, DaskVector):
            raise TypeError("range should not a DaskVector object!")
        if not isinstance(range, vector):
            raise TypeError("range is not a vector-derived object!")
        if domain.size != range.size:
            raise ValueError("number of elements in domain and range is not equal!")
        super(DaskCollectOp, self).__init__(domain, range)

    def forward(self, add, model, data):
        """Forward operator: collecting dask vector array to local one"""
        if not isinstance(model, DaskVector):
            raise TypeError("Model vector must be a DaskVector!")
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        dataNd = data.getNdArray().ravel()
        # Obtaining remote array/s
        modelNd_list = model.getNdArray()
        idx = 0
        # Adding remote arrays to local one
        for arr in modelNd_list:
            n_elem = arr.size
            dataNd[idx:idx + n_elem] += arr.ravel()
            idx += n_elem
        return

    def adjoint(self, add, model, data):
        """Adjoint operator: scattering/distributing local array to remote vector"""
        if not isinstance(model, DaskVector):
            raise TypeError("Model vector must be a DaskVector!")
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        dataNd = data.getNdArray().ravel()
        # Shapes of the DaskVector chunks
        shapes = model.shape
        # Scattering local array
        client = model.client
        idx_el = 0
        for idx, shape in enumerate(shapes):
            n_elem = np.prod(shape)
            wrkId = list(client.who_has(model.vecDask[idx]).values())[0]
            arrD = client.scatter(dataNd[idx_el:idx_el + n_elem], workers=wrkId)
            daskD.wait(arrD)
            daskD.wait(client.submit(add_from_NdArray, model.vecDask[idx], arrD, pure=False))
            idx_el += n_elem
        return
