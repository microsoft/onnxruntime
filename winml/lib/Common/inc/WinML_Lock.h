#pragma once

//
//  Simple CRITICAL_SECTION based locks
//
class CWinML_Lock
{
private:
    // make copy constructor and assignment operator inaccessible

    CWinML_Lock(const CWinML_Lock &refCritSec);
    CWinML_Lock &operator=(const CWinML_Lock &refCritSec);

    CRITICAL_SECTION m_CritSec;

public:
    CWinML_Lock() {
        InitializeCriticalSection(&m_CritSec);
    };

    ~CWinML_Lock() {
        DeleteCriticalSection(&m_CritSec);
    };

    void Lock() {
        EnterCriticalSection(&m_CritSec);
    };
    void Unlock() {
        LeaveCriticalSection(&m_CritSec);
    };
    void LockExclusive() {
        EnterCriticalSection(&m_CritSec);
    };
    void UnlockExclusive() {
        LeaveCriticalSection(&m_CritSec);
    };
    BOOL IsLockHeldByCurrentThread()
    {
        return GetCurrentThreadId() == static_cast<DWORD>(reinterpret_cast<ULONG_PTR>(m_CritSec.OwningThread));
    };
    BOOL IsLockHeld()
    {
        return m_CritSec.OwningThread != 0;
    };
    BOOL TryLock()
    {
        return TryEnterCriticalSection(&m_CritSec);
    };
    // aliased methods to help code compat so that CriticalSections can be passed to ReaderWriter templates
    void LockShared() {
        EnterCriticalSection(&m_CritSec);
    };
    void UnlockShared() {
        LeaveCriticalSection(&m_CritSec);
    };
};


// locks a critical section, and unlocks it automatically
// when the lock goes out of scope
class CWinML_AutoLock {

    // make copy constructor and assignment operator inaccessible

    CWinML_AutoLock(const CWinML_AutoLock &refAutoLock);
    CWinML_AutoLock &operator=(const CWinML_AutoLock &refAutoLock);

protected:
    CWinML_Lock * m_pLock;

public:
    CWinML_AutoLock(CWinML_Lock * plock)
    {
        m_pLock = plock;
        if (m_pLock != nullptr)
        {
            m_pLock->Lock();
        }
    };

    ~CWinML_AutoLock() 
    {
        if (m_pLock != nullptr)
        {
            m_pLock->Unlock();
        }
    };
};
