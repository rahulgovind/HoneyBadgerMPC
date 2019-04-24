from libcpp.vector cimport vector

cdef extern from "ntlwrapper.h":
    cdef cppclass ZZ "ZZ":
        ZZ() except +
        pass

    cdef cppclass ZZ_pContext_c "ZZ_pContext":
        void restore()
        pass

    cdef cppclass ZZ_p "ZZ_p":
        ZZ_p() except +

    cdef cppclass ZZ_pX_c "ZZ_pX":
        void SetMaxLength(size_t)
        pass

    cdef cppclass vec_ZZ_p:
        ZZ_p& operator[](size_t)
        void SetLength(int n) nogil
        void kill()
        size_t length()
    cdef cppclass mat_ZZ_p:
        void SetDims(int m, int n)
        vec_ZZ_p& operator[](size_t)
        void kill()


    void ZZ_p_init "ZZ_p::init"(ZZ x) nogil
    void SetNumThreads(int n)
    void ZZonv_from_int "conv"(ZZ x, int i)
    void ZZ_ponv_from_int "conv"(ZZ_p x, int i)
    void ZZonv_to_int "conv"(int i, ZZ x)
    void ZZonv_from_long "conv"(ZZ x, long l)
    void ZZonv_to_long "conv"(long l, ZZ x)
    void mat_ZZ_p_mul "mul"(mat_ZZ_p x, mat_ZZ_p a, mat_ZZ_p b)
    void mat_ZZ_p_mul_vec "mul"(vec_ZZ_p x, mat_ZZ_p a, vec_ZZ_p b) nogil
    void ZZ_pX_get_coeff "GetCoeff"(ZZ_p r, ZZ_pX_c x, int i)
    void ZZ_pX_set_coeff "SetCoeff"(ZZ_pX_c x, int i, ZZ_p a)
    void ZZ_pX_eval "eval" (ZZ_p b, ZZ_pX_c f, ZZ_p a)
    void SqrRootMod "SqrRootMod"(ZZ x, ZZ a, ZZ n)
    int AvailableThreads()
    ZZ ZZFromBytes(const unsigned char*, long)
    unsigned char* bytesFromZZ(ZZ x)
    ZZ_p to_ZZ_p(ZZ)
    ZZ to_ZZ "rep"(ZZ_p)
    int ZZNumBytes "NumBytes"(ZZ)