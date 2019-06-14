import numpy as np
import logging
import bet.util as util
import bet.sample as samp
import bet.sampling.basicSampling as bsam
import scipy.spatial.distance as ds


def density(sample_set, ptr=None):
    r"""
    Compute density for a sample set and write it to the ``_emulated_density``
    attribute inside of ``sample_set``

    :param sample_set: sample set with existing probabilities stored
    :type sample_set: :class:`bet.sample.sample_set_base`
    :param ptr: pointer to a reference set against which densities are
        being compared. If ``None``, use samples as they are.
    :type ptr: list, tuple, or ``np.ndarray``

    :rtype: :class:`bet.sample.sample_set_base`
    :returns: sample set object with additional attribute ``_emulated_density``

    """
    if sample_set is None:
        raise AttributeError("Missing sample set.")
    elif hasattr(sample_set, '_density'):
        # this is our way of checking if we used sampling-approach
        # if already computed, avoid re-computation.
        if ptr is not None:
            den = sample_set._density[ptr]
        else:
            den = sample_set._density
        sample_set._emulated_density = den
    else:  # not none
        if sample_set._probabilities is None:
            raise AttributeError("Missing probabilities from sample set.")
        if sample_set._volumes is None:
            raise AttributeError("Missing volumes from sample set.")
        if sample_set._probabilities_local is None:
            sample_set.global_to_local()

        if ptr is None:
            den = np.divide(sample_set._probabilities.ravel(),
                            sample_set._volumes.ravel())
        else:
            den = np.divide(sample_set._probabilities[ptr].ravel(),
                            sample_set._volumes[ptr].ravel())
        sample_set._emulated_density = den
    if ptr is None:  # create pointer to density to avoid re-run
        sample_set._density = sample_set._emulated_density
    else:
        sample_set._prob = sample_set._probabilities[ptr].ravel()
    sample_set.local_to_global()
    return sample_set


def compare(left_set, right_set, num_mc_points=1000):
    r"""
    This is a convience function to quickly instantiate and return
    a `~bet.postProcess.comparison` object. See the docstring for
    this class for more details.

    :param left set: sample set in left position
    :type left set: :class:`bet.sample.sample_set_base`
    :param right set: sample set in right position
    :type right set: :class:`bet.sample.sample_set_base`
    :param int num_mc_points: number of values of sample set to return

    :rtype: :class:`~bet.postProcess.compareP.comparison`
    :returns: comparison object

    """
    # extract sample set
    if isinstance(left_set, samp.discretization):
        logging.log(20, "Discretization passed. Assuming input set.")
        left_set = left_set.get_input_sample_set()
    if isinstance(right_set, samp.discretization):
        logging.log(20, "Discretization passed. Assuming input set.")
        right_set = right_set.get_input_sample_set()
    if not num_mc_points > 0:
        raise ValueError("Please specify positive num_mc_points")

    # make integration sample set
    assert left_set.get_dim() == right_set.get_dim()
    assert np.array_equal(left_set.get_domain(), right_set.get_domain())
    em_set = samp.sample_set(left_set.get_dim())
    em_set.set_domain(right_set.get_domain())
    em_set = bsam.random_sample_set('r', em_set, num_mc_points)

    # to be generating a new random sample set pass an integer argument
    comp = comparison(em_set, left_set, right_set)

    return comp


class comparison(object):
    """
    This class allows for analytically-sound comparisons between
    probability measures defined on different sigma-algebras. In order
    to compare the similarity of two measures defined on different
    sigma-algebras (induced by the voronoi-cell tesselations implicitly
    defined by the ``_values`` in each sample set), a third sample set
    object is introduced as a reference for comparison. It is referred
    to as an ``emulated_sample_set`` and is required to instantiate a
    ``comparison`` object since the dimensions will be used to enforce
    properly setting the left and right sample set positions.

    This object can be thought of as a more flexible version of an abstraction
    of a metric, a measure of distance between two probability measures.
    A metric ``d(x,y)`` has two arguments, one to the left (``x``),
    and one to the right (``y``). However, we do not enforce the properties
    that define a formal metric, instead we use the language of "comparisons".

    Technically, any function can be passed for evaluation, including
    ones that fail to satisfy symmetry, so we refrain from reffering
    to measures of similarity as metrics, though this is the usual case
    (with the exception of the frequently used KL-Divergence).
    Several common measures of similarity are accessible with keywords.

    The number of samples in this third (reference) sample set is
    given by the argument ``num_mc_points``, and pointers between this
    set and the left/right sets are built on-demand. Methods in this
    class allow for over-writing of any of the three sample set objects
    involved, and pointers are re-built either by explictly, or they
    are computed when the a measure of similarity (such as distance) is
    requested to be evaluated.

    .. seealso::

        :meth:`bet.compareP.comparison.value``

    :param emulated_sample_set: Reference set against which comparisons
        will be made.
    :type emulated_sample_set: :class:`bet.sample.sample_set_base`

    """
    #: List of attribute names for attributes which are vectors or 1D
    #: :class:`numpy.ndarray`
    vector_names = ['_ptr_left', '_ptr_left_local',
                    '_ptr_right', '_ptr_right_local', '_domain']

    #: List of attribute names for attributes that are
    #: :class:`sample.sample_set_base`
    sample_set_names = ['_sample_set_left', '_sample_set_right',
                        '_emulated_sample_set']

    def __init__(self, emulated_sample_set,
                 sample_set_left=None, sample_set_right=None,
                 ptr_left=None, ptr_right=None):
        #: Left sample set
        self._sample_set_left = None
        #: Right sample set
        self._sample_set_right = None
        #: Integration/Emulation set :class:`~bet.sample.sample_set_base`
        self._emulated_sample_set = emulated_sample_set
        #: Pointer from ``self._emulated_sample_set`` to
        #: ``self._sample_set_left``
        self._ptr_left = ptr_left
        #: Pointer from ``self._emulated_sample_set`` to
        #: ``self._sample_set_right``
        self._ptr_right = ptr_right
        #: local integration left ptr for parallelsim
        self._ptr_left_local = None
        #: local integration right ptr for parallelism
        self._ptr_right_local = None
        #: Domain
        self._domain = None
        #: Left sample set density evaluated on emulation set.
        self._den_left = None
        #: Right sample set density evaluated on emulation set.
        self._den_right = None

        # extract sample set
        if isinstance(sample_set_left, samp.sample_set_base):
            # left sample set
            self._sample_set_left = sample_set_left
            self._domain = sample_set_left.get_domain()
        if isinstance(sample_set_right, samp.sample_set_base):
            # right sample set
            self._sample_set_right = sample_set_right
            if self._domain is not None:
                if not np.allclose(self._domain, sample_set_right._domain):
                    raise samp.domain_not_matching(
                        "Left and Right domains do not match")
            else:
                self._domain = sample_set_right.get_domain()

        # check dimension consistency
        if isinstance(emulated_sample_set, samp.sample_set_base):
            self._num_samples = emulated_sample_set.check_num()
            output_dims = []
            output_dims.append(emulated_sample_set.get_dim())
            if self._sample_set_right is not None:
                output_dims.append(self._sample_set_right.get_dim())
            if self._sample_set_left is not None:
                output_dims.append(self._sample_set_left.get_dim())
            if len(output_dims) == 1:
                self._emulated_sample_set = emulated_sample_set
            elif np.all(np.array(output_dims) == output_dims[0]):
                self._emulated_sample_set = emulated_sample_set
            else:
                raise samp.dim_not_matching("dimension of values incorrect")

            if not isinstance(emulated_sample_set.get_domain(), np.ndarray):
                # domain can be missing if left/right sample sets present
                if self._sample_set_left is not None:
                    emulated_sample_set.set_domain(self._domain)
                else:
                    if self._sample_set_right is not None:
                        emulated_sample_set.set_domain(self._domain)
                    else:  # no sample sets provided
                        msg = "Must provide at least one set from\n"
                        msg += "\twhich a domain can be inferred."
                        raise AttributeError(msg)
        else:
            if (self._sample_set_left is not None) or \
               (self._sample_set_right is not None):
                pass
            else:
                raise AttributeError(
                    "Wrong Type: Should be samp.sample_set_base type")

        if (ptr_left is not None):
            if len(ptr_left) != self._num_samples:
                raise AttributeError(
                    "Left pointer length must match integration set.")
            if (ptr_right is not None):
                if not np.allclose(ptr_left.shape, ptr_right.shape):
                    raise AttributeError("Pointers must be of same length.")
        if (ptr_right is not None):
            if len(ptr_right) != self._num_samples:
                raise AttributeError(
                    "Right pointer length must match integration set.")

    def check_dim(self):
        r"""
        Checks that dimensions of left and right sample sets match
        the dimension of the emulated sample set.

        :rtype: int
        :returns: dimension

        """
        left_set = self.get_left()
        right_set = self.get_right()
        if left_set.get_dim() != right_set.get_dim():
            msg = "These sample sets must have the same dimension."
            raise samp.dim_not_matching(msg)
        else:
            dim = left_set.get_dim()

        il, ir = self.get_ptr_left(), self.get_ptr_right()
        if (il is not None) and (ir is not None):
            if len(il) != len(ir):
                msg = "The pointers have inconsistent sizes."
                msg += "\nTry running set_ptr_left() [or _right()]"
                raise samp.dim_not_matching(msg)
        return dim

    def check_domain(self):
        r"""
        Checks that all domains match so that the comparisons
        are being made on measures defined on the same underlying space.

        :rtype: ``np.ndarray`` of shape (ndim, 2)
        :returns: domain bounds

        """
        left_set = self.get_left()
        right_set = self.get_right()
        if left_set._domain is not None and right_set._domain is not None:
            if not np.allclose(left_set._domain, right_set._domain):
                msg = "These sample sets have different domains."
                raise samp.domain_not_matching(msg)
            else:
                domain = left_set.get_domain()
        else:  # since the domains match, we can choose either.
            if left_set._domain is None or right_set._domain is None:
                msg = "One or more of your sets is missing a domain."
                raise samp.domain_not_matching(msg)

        if not np.allclose(self._emulated_sample_set.get_domain(), domain):
            msg = "Integration domain mismatch."
            raise samp.domain_not_matching(msg)
        self._domain = domain
        return domain

    def globalize_ptrs(self):
        r"""
        Globalizes comparison pointers by caling ``get_global_values``
        for both the left and right sample sets.

        """
        if (self._ptr_left_local is not None) and\
                (self._ptr_left is None):
            self._ptr_left = util.get_global_values(
                self._ptr_left_local)
        if (self._ptr_right_local is not None) and\
                (self._ptr_right is None):
            self._ptr_right = util.get_global_values(
                self._ptr_right_local)

    def set_ptr_left(self, globalize=True):
        """
        Creates the pointer from ``self._emulated_sample_set`` to
        ``self._sample_set_left``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :param bool globalize: flag whether or not to globalize
            ``self._ptr_left``

        """
        if self._emulated_sample_set._values_local is None:
            self._emulated_sample_set.global_to_local()

        (_, self._ptr_left_local) = self._sample_set_left.query(
            self._emulated_sample_set._values_local)

        if globalize:
            self._ptr_left = util.get_global_values(
                self._ptr_left_local)
        assert self._sample_set_left.check_num() >= max(self._ptr_left_local)

    def get_ptr_left(self):
        """
        Returns the pointer from ``self._emulated_sample_set`` to
        ``self._sample_set_left``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :rtype: :class:`numpy.ndarray` of int of shape
            (self._sample_set_left._values.shape[0],)
        :returns: self._ptr_left

        """
        return self._ptr_left

    def set_ptr_right(self, globalize=True):
        """
        Creates the pointer from ``self._emulated_sample_set`` to
        ``self._sample_set_right``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :param bool globalize: flag whether or not to globalize
            ``self._ptr_right``

        """
        if self._emulated_sample_set._values_local is None:
            self._emulated_sample_set.global_to_local()

        (_, self._ptr_right_local) = self._sample_set_right.query(
            self._emulated_sample_set._values_local)

        if globalize:
            self._ptr_right = util.get_global_values(
                self._ptr_right_local)
        assert self._sample_set_right.check_num() >= max(self._ptr_right_local)

    def get_ptr_right(self):
        """
        Returns the pointer from ``self._emulated_sample_set`` to
        ``self._sample_set_right``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :rtype: :class:`numpy.ndarray` of int of shape
            (self._sample_set_right._values.shape[0],)
        :returns: self._ptr_right

        """
        return self._ptr_right

    def copy(self):
        """
        Makes a copy using :meth:`numpy.copy`.

        :rtype: :class:`~bet.postProcess.compareP.comparison`
        :returns: Copy of a comparison object.

        """
        my_copy = comparison(self._emulated_sample_set.copy(),
                             self._sample_set_left.copy(),
                             self._sample_set_right.copy())

        for attrname in comparison.sample_set_names:
            if attrname is not '_sample_set_left' and \
                    attrname is not '_sample_set_right':
                curr_sample_set = getattr(self, attrname)
                if curr_sample_set is not None:
                    setattr(my_copy, attrname, curr_sample_set.copy())

        for array_name in comparison.vector_names:
            current_array = getattr(self, array_name)
            if current_array is not None:
                setattr(my_copy, array_name, np.copy(current_array))
        return my_copy

    def get_sample_set_left(self):
        """
        Returns a reference to the left sample set for this comparison.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: left sample set

        """
        return self._sample_set_left

    def get_left(self):
        r"""
        Wrapper for `get_sample_set_left`.
        """
        return self.get_sample_set_left()

    def set_sample_set_left(self, sample_set):
        """

        Sets the left sample set for this comparison.

        :param sample_set: left sample set
        :type sample_set: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(sample_set, samp.sample_set_base):
            self._sample_set_left = sample_set
            self._ptr_left = None
            self._ptr_left_local = None
            self._den_left = None
        elif isinstance(sample_set, samp.discretization):
            logging.log(20, "Discretization passed. Assuming input set.")
            sample_set = sample_set.get_input_sample_set()
            self._sample_set_left = sample_set
            self._ptr_left = None
            self._ptr_left_local = None
            self._den_left = None
        else:
            raise TypeError(
                "Wrong Type: Should be samp.sample_set_base type")
        if self._emulated_sample_set._domain is None:
            self._emulated_sample_set.set_domain(
                sample_set.get_domain())
        else:
            if not np.allclose(self._emulated_sample_set._domain,
                               sample_set._domain):
                raise samp.domain_not_matching(
                    "Domain does not match integration set.")

    def set_left(self, sample_set):
        r"""

        Wrapper for `set_sample_set_left`.

        :param sample_set: sample set
        :type sample_set: :class:`~bet.sample.sample_set_base`

        """
        return self.set_sample_set_left(sample_set)

    def get_sample_set_right(self):
        """

        Returns a reference to the right sample set for this comparison.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: right sample set

        """
        return self._sample_set_right

    def get_right(self):
        r"""
        Wrapper for `get_sample_set_right`.
        """
        return self.get_sample_set_right()

    def set_right(self, sample_set):
        r"""

        Wrapper for `set_sample_set_right`.

        :param sample_set: sample set
        :type sample_set: :class:`~bet.sample.sample_set_base`

        """
        return self.set_sample_set_right(sample_set)

    def set_sample_set_right(self, sample_set):
        """
        Sets the right sample set for this comparison.

        :param sample_set: right sample set
        :type sample_set: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(sample_set, samp.sample_set_base):
            self._sample_set_right = sample_set
            self._ptr_right = None
            self._ptr_right_local = None
            self._den_right = None
        elif isinstance(sample_set, samp.discretization):
            logging.log(20, "Discretization passed. Assuming input set.")
            sample_set = sample_set.get_input_sample_set()
            self._sample_set_right = sample_set
            self._ptr_right = None
            self._ptr_right_local = None
            self._den_right = None
        else:
            raise TypeError(
                "Wrong Type: Should be samp.sample_set_base type")
        if self._emulated_sample_set._domain is None:
            self._emulated_sample_set.set_domain(
                sample_set.get_domain())
        else:
            if not np.allclose(self._emulated_sample_set._domain,
                               sample_set._domain):
                raise samp.domain_not_matching(
                    "Domain does not match integration set.")

    def get_emulated_sample_set(self):
        r"""
        Returns a reference to the emulated sample set for this comparison.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: emulated sample set

        """
        return self._emulated_sample_set

    def get_emulated(self):
        r"""
        Wrapper for `get_emulated_sample_set`.
        """
        return self.get_emulated_sample_set()

    def set_emulated_sample_set(self, emulated_sample_set):
        r"""
        Sets the emulated sample set for this comparison.

        :param emulated_sample_set: emulated sample set
        :type emulated_sample_set: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(emulated_sample_set, samp.sample_set_base):
            output_dims = []
            output_dims.append(emulated_sample_set.get_dim())
            if self._sample_set_right is not None:
                output_dims.append(self._sample_set_right.get_dim())
            if self._sample_set_left is not None:
                output_dims.append(self._sample_set_left.get_dim())
            if len(output_dims) == 1:
                self._emulated_sample_set = emulated_sample_set
            elif np.all(np.array(output_dims) == output_dims[0]):
                self._emulated_sample_set = emulated_sample_set
            else:
                raise samp.dim_not_matching("dimension of values incorrect")
        else:
            raise AttributeError(
                "Wrong Type: Should be samp.sample_set_base type")
        # if a new emulation set is provided, forget the emulated evaluation.
        if self._sample_set_left is not None:
            self._sample_set_left._emulated_density = None
        if self._sample_set_right is not None:
            self._sample_set_right._emulated_density = None

    def set_emulated(self, sample_set):
        r"""
        Wrapper for `set_emulated_sample_set`.

        :param sample_set: sample set
        :type sample_set: :class:`~bet.sample.sample_set_base`

        """
        return self.set_emulated_sample_set(sample_set)

    def clip(self, lnum, rnum=None, copy=True):
        r"""
        Creates and returns a comparison with the the first `lnum`
        and `rnum` entries of the left and right sample sets, respectively.

        :param int lnum: number of values in left sample set to return.
        :param int rnum: number of values in right sample set to return.
            If ``rnum==None``, set ``rnum=lnum``.
        :param bool copy: Pass emulated_sample_set by value instead of pass
            by reference (use same pointer to sample set object).

        :rtype: :class:`~bet.sample.comparison`
        :returns: clipped comparison

        """
        if rnum is None:  # can clip by same amount
            rnum = lnum
        if lnum > 0:
            cl = self._sample_set_left.clip(lnum)
        else:
            cl = self._sample_set_left.copy()
        if rnum > 0:
            cr = self._sample_set_right.clip(rnum)
        else:
            cr = self._sample_set_right.copy()

        if copy:
            em_set = self._emulated_sample_set.copy()
        else:
            em_set = self._emulated_sample_set

        return comparison(sample_set_left=cl,
                          sample_set_right=cr,
                          emulated_sample_set=em_set)

    def merge(self, comp):
        r"""
        Merges a given comparison with this one by merging the input and
        output sample sets.

        :param comp: comparison object to merge with.
        :type comp: :class:`bet.sample.comparison`

        :rtype: :class:`bet.sample.comparison`
        :returns: Merged comparison
        """
        ml = self._sample_set_left.merge(comp._sample_set_left)
        mr = self._sample_set_right.merge(comp._sample_set_right)
        il, ir = self._ptr_left, self._ptr_right
        if comp._ptr_left is not None:
            il += comp._ptr_left
        if comp._ptr_right is not None:
            ir += comp._ptr_right
        return comparison(sample_set_left=ml,
                          sample_set_right=mr,
                          emulated_sample_set=self._emulated_sample_set,
                          ptr_left=il,
                          ptr_right=ir)

    def slice(self,
              dims=None):
        r"""
        Slices the left and right of the comparison.

        :param list dims: list of indices (dimensions) of sample set to include

        :rtype: :class:`~bet.sample.comparison`
        :returns: sliced comparison

        """
        slice_list = ['_values', '_values_local',
                      '_error_estimates', '_error_estimates_local',
                      ]
        slice_list2 = ['_jacobians', '_jacobians_local']

        int_ss = samp.sample_set(len(dims))
        left_ss = samp.sample_set(len(dims))
        right_ss = samp.sample_set(len(dims))

        if self._emulated_sample_set._domain is not None:
            int_ss.set_domain(self._emulated_sample_set._domain[dims, :])

        if self._sample_set_left._domain is not None:
            left_ss.set_domain(self._sample_set_left._domain[dims, :])
        if self._sample_set_left._reference_value is not None:
            left_ss.set_reference_value(
                self._sample_set_left._reference_value[dims])

        if self._sample_set_right._domain is not None:
            right_ss.set_domain(self._sample_set_right._domain[dims, :])
        if self._sample_set_right._reference_value is not None:
            right_ss.set_reference_value(
                self._sample_set_right._reference_value[dims])

        for obj in slice_list:
            val = getattr(self._sample_set_left, obj)
            if val is not None:
                setattr(left_ss, obj, val[:, dims])
            val = getattr(self._sample_set_right, obj)
            if val is not None:
                setattr(right_ss, obj, val[:, dims])
            val = getattr(self._emulated_sample_set, obj)
            if val is not None:
                setattr(int_ss, obj, val[:, dims])
        for obj in slice_list2:
            val = getattr(self._sample_set_left, obj)
            if val is not None:
                nval = np.copy(val)
                nval = nval.take(dims, axis=1)
                nval = nval.take(dims, axis=2)
                setattr(left_ss, obj, nval)
            val = getattr(self._sample_set_right, obj)
            if val is not None:
                nval = np.copy(val)
                nval = nval.take(dims, axis=1)
                nval = nval.take(dims, axis=2)
                setattr(right_ss, obj, nval)

        comp = comparison(sample_set_left=left_ss,
                          sample_set_right=right_ss,
                          emulated_sample_set=int_ss)
        # additional attributes to copy over here. TODO: maybe slice through
        return comp

    def global_to_local(self):
        """
        Call global_to_local for ``sample_set_left`` and
        ``sample_set_right``.

        """
        if self._sample_set_left is not None:
            self._sample_set_left.global_to_local()
        if self._sample_set_right is not None:
            self._sample_set_right.global_to_local()
        if self._emulated_sample_set is not None:
            self._emulated_sample_set.global_to_local()

    def local_to_global(self):
        """
        Call local_to_global for ``sample_set_left``,
        ``sample_set_right``, and ``emulated_sample_set``.

        """
        if self._sample_set_left is not None:
            self._sample_set_left.local_to_global()
        if self._sample_set_right is not None:
            self._sample_set_right.local_to_global()
        if self._emulated_sample_set is not None:
            self._emulated_sample_set.local_to_global()

    def estimate_volume_mc(self):
        r"""
        Applies MC assumption to volumes of both sets.
        """
        self._sample_set_left.estimate_volume_mc()
        self._sample_set_right.estimate_volume_mc()

    def set_left_probabilities(self, probabilities):
        r"""
        Allow overwriting of probabilities for the left sample set.

        :param probabilities: probabilities to overwrite the ones in the
            left sample set.
        :type probabilities: list, tuple, or `numpy.ndarray`

        """
        if self.get_left().check_num() != len(probabilities):
            raise AttributeError("Length of probabilities incorrect.")
        self._sample_set_left.set_probabilities(probabilities)
        self._sample_set_left.global_to_local()
        self._sample_set_left._emulated_density = None
        self._den_left = None

    def set_right_probabilities(self, probabilities):
        r"""
        Allow overwriting of probabilities for the right sample set.

        :param probabilities: probabilities to overwrite the ones in the
            right sample set.
        :type probabilities: list, tuple, or `numpy.ndarray`

        """
        if self.get_right().check_num() != len(probabilities):
            raise AttributeError("Length of probabilities incorrect.")
        self._sample_set_right._probabilities = probabilities
        self._sample_set_right.global_to_local()
        self._sample_set_right._emulated_density = None
        self._den_right = None

    def get_left_probabilities(self):
        r"""
        Wrapper for ``get_probabilities`` for the left sample set.
        """
        return self._sample_set_left.get_probabilities()

    def get_right_probabilities(self):
        r"""
        Wrapper for ``get_probabilities`` for the right sample set.
        """
        return self._sample_set_right.get_probabilities()

    def set_volume_emulated(self, sample_set, emulated_sample_set=None):
        r"""
        Wrapper to use the emulated sample set for the
        calculation of volumes on the sample sets (as opposed to using the
        Monte-Carlo assumption or setting volumes manually.)

        .. seealso::

            :meth:`bet.compareP.comparison.estimate_volume_mc``
            :meth:`bet.compareP.comparison.set_left_volume_emulated``
            :meth:`bet.compareP.comparison.set_right_volume_emulated``

        :param sample_set: sample set
        :type sample_set: :class:`~bet.sample.sample_set_base`
        :param emulated_sample_set: emulated sample set
        :type emulated_sample_set: :class:`~bet.sample.sample_set_base`


        """
        if emulated_sample_set is not None:
            if not isinstance(emulated_sample_set, samp.sample_set_base):
                msg = "Wrong type specified for `emulation_set`.\n"
                msg += "Please specify a `~bet.sample.sample_set_base`."
                raise AttributeError(msg)
            else:
                sample_set.estimate_volume_emulated(emulated_sample_set)
        else:
            # if not defined, use existing emulated set for volumes.
            sample_set.estimate_volume_emulated(self._emulated_sample_set)

    def set_left_volume_emulated(self, emulated_sample_set=None):
        r"""
        Use an emulated sample set to define volumes for the left set.
        """
        self.set_volume_emulated(self.get_left(), emulated_sample_set)
        self._den_left = None  # if volumes change, so will densities.

    def set_right_volume_emulated(self, emulated_sample_set=None):
        r"""
        Use an emulated sample set to define volumes for the right set.

        :param emulated_sample_set: emulated sample set
        :type emulated_sample_set: :class:`~bet.sample.sample_set_base`

        """
        self.set_volume_emulated(self.get_right(), emulated_sample_set)
        self._den_right = None  # if volumes change, so will densities.

    def estimate_density_left(self):
        r"""
        Evaluates density function for the left probability measure
        at the set of samples defined in `emulated_sample_set`.

        """
        s_set = self.get_left()
        if self._ptr_left_local is None:
            self.set_ptr_left()
        s_set = density(s_set, self._ptr_left_local)
        self._den_left = s_set._emulated_density
        return self._den_left

    def estimate_density_right(self):
        r"""
        Evaluates density function for the right probability measure
        at the set of samples defined in ``emulated_sample_set``.

        """
        s_set = self.get_right()
        if self._ptr_right_local is None:
            self.set_ptr_right()
        s_set = density(s_set, self._ptr_right_local)
        self._den_right = s_set._emulated_density
        return self._den_right

    def estimate_right_density(self):
        r"""
        Wrapper for ``bet.postProcess.compareP.estimate_density_right``.
        """
        return self.estimate_density_right()

    def estimate_left_density(self):
        r"""
        Wrapper for ``bet.postProcess.compareP.estimate_density_left``.
        """
        return self.estimate_density_left()

    def get_density_right(self):
        r"""
        Returns right emulated density.
        """
        return self._den_right

    def get_density_left(self):
        r"""
        Returns left emulated density.
        """
        return self._den_left

    def get_left_density(self):
        r"""
        Wrapper for ``bet.postProcess.compareP.get_density_left``.
        """
        return self.get_density_left()

    def get_right_density(self):
        r"""
        Wrapper for ``bet.postProcess.compareP.get_density_right``.
        """
        return self.get_density_right()

    def estimate_density(self, globalize=True,
                         emulated_sample_set=None):
        r"""
        Evaluate density functions for both left and right sets using
        the set of samples defined in ``self._emulated_sample_set``.

        :param bool globalize: globalize left/right sample sets
        :param emulated_sample_set: emulated sample set
        :type emulated_sample_set: :class:`~bet.sample.sample_set_base`

        :rtype: ``numpy.ndarray``, ``numpy.ndarray``
        :returns: left and right density values

        """
        if globalize:  # in case probabilities were re-set but not local
            self.global_to_local()

        em_set = self.get_emulated_sample_set()
        if em_set is None:
            raise AttributeError("Missing integration set.")
        self.check_domain()

        # set pointers if they have not already been set
        if self._ptr_left_local is None:
            self.set_ptr_left(globalize)
        if self._ptr_right_local is None:
            self.set_ptr_right(globalize)
        self.check_dim()

        left_set, right_set = self.get_left(), self.get_right()

        if left_set._volumes is None:
            if emulated_sample_set is None:
                msg = " Volumes missing from left. Using MC assumption."
                logging.log(20, msg)
                left_set.estimate_volume_mc()
            else:
                self.set_left_volume_emulated(emulated_sample_set)
        else:  # volumes present and emulated passed
            if emulated_sample_set is not None:
                msg = " Overwriting left volumes with emulated ones."
                logging.log(20, msg)
                self.set_left_volume_emulated(emulated_sample_set)

        if right_set._volumes is None:
            if emulated_sample_set is None:
                msg = " Volumes missing from right. Using MC assumption."
                logging.log(20, msg)
                right_set.estimate_volume_mc()
            else:
                msg = " Overwriting right volumes with emulated ones."
                logging.log(20, msg)
                self.set_right_volume_emulated(emulated_sample_set)
        else:  # volumes present and emulated passed
            if emulated_sample_set is not None:
                self.set_right_volume_emulated(emulated_sample_set)

        # compute densities
        self.estimate_density_left()
        self.estimate_density_right()

        if globalize:
            self.local_to_global()
        return self._den_left, self._den_right

    def value(self, functional='tv', **kwargs):
        r"""
        Compute value capturing some meaure of similarity using the
        evaluated densities on a shared emulated set.
        If either density evaluation is missing, re-compute it.

        :param funtional: a function representing a measure of similarity
        :type functional: method that takes in two lists/arrays and returns
            a scalar value (measure of similarity)

        :rtype: float
        :returns: value representing a measurement between the left and right
            sample sets, ideally a measure of similarity, a distance, a metric.

        """
        left_den, right_den = self.get_left_density(), self.get_right_density()
        if left_den is None:
            # logging.log(20,"Left density missing. Estimating now.")
            left_den = self.estimate_density_left()
        if right_den is None:
            # logging.log(20,"Right density missing. Estimating now.")
            right_den = self.estimate_density_right()

        if functional in ['tv', 'totvar',
                          'total variation', 'total-variation', '1']:
            dist = ds.minkowski(left_den, right_den, 1, w=0.5, **kwargs)
        elif functional in ['mink', 'minkowski']:
            dist = ds.minkowski(left_den, right_den, **kwargs)
        elif functional in ['norm']:
            dist = ds.norm(left_den - right_den, **kwargs)
        elif functional in ['euclidean', '2-norm', '2']:
            dist = ds.minkowski(left_den, right_den, 2, **kwargs)
        elif functional in ['sqhell', 'sqhellinger']:
            dist = ds.sqeuclidean(np.sqrt(left_den), np.sqrt(right_den)) / 2.0
        elif functional in ['hell', 'hellinger']:
            return np.sqrt(self.value('sqhell'))
        else:
            dist = functional(left_den, right_den, **kwargs)

        return dist / self._emulated_sample_set.check_num()