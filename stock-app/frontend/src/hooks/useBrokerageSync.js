/**
 * useBrokerageSync — shared state for all brokerage sync UI.
 * Consumed by BrokerageManager and Portfolio so state is shared across both.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';

export function useBrokerageSync(options = {}) {
    const { lazy = false } = options;
    const startedRef = useRef(false);
    const [isStarted, setIsStarted] = useState(false);
    const [accounts, setAccounts]   = useState([]);
    const [positions, setPositions] = useState([]);
    const [balances, setBalances]   = useState([]);
    const [summary, setSummary]     = useState(null);
    const [isSyncing, setIsSyncing] = useState(false);
    const [syncingAccountId, setSyncingAccountId] = useState(null); // tracks per-account spinner
    const [isRegistered, setIsRegistered] = useState(null); // null = unknown, true/false
    const pollRef = useRef(null);

    // ------------------------------------------------------------------
    // Registration check / first-time setup
    // ------------------------------------------------------------------
    const ensureRegistered = useCallback(async () => {
        try {
            const res = await axios.post('/api/brokerage/register');
            setIsRegistered(true);
            return res.data;
        } catch (err) {
            if (err.response?.status === 404) {
                setIsRegistered(false);
            }
            throw err;
        }
    }, []);

    // ------------------------------------------------------------------
    // Data fetchers
    // ------------------------------------------------------------------
    const fetchAccounts = useCallback(async () => {
        try {
            const res = await axios.get('/api/brokerage/accounts');
            setAccounts(res.data);
            setIsRegistered(true);
        } catch (err) {
            if (err.response?.status === 404) {
                setIsRegistered(false);
                setAccounts([]);
            }
        }
    }, []);

    const fetchPositions = useCallback(async () => {
        try {
            const res = await axios.get('/api/brokerage/positions');
            setPositions(res.data);
        } catch {
            setPositions([]);
        }
    }, []);

    const fetchBalances = useCallback(async () => {
        try {
            const res = await axios.get('/api/brokerage/balances');
            setBalances(res.data);
        } catch {
            setBalances([]);
        }
    }, []);

    const fetchSummary = useCallback(async () => {
        try {
            const res = await axios.get('/api/brokerage/summary');
            setSummary(res.data);
        } catch {
            setSummary(null);
        }
    }, []);

    const fetchAll = useCallback(async () => {
        await Promise.all([fetchAccounts(), fetchPositions(), fetchBalances(), fetchSummary()]);
    }, [fetchAccounts, fetchPositions, fetchBalances, fetchSummary]);

    const start = useCallback(async () => {
        if (startedRef.current) return;
        startedRef.current = true;
        setIsStarted(true);
        try {
            await ensureRegistered();
        } catch {
            // Not registered yet — will show prompt
        }
        await fetchAll();
    }, [ensureRegistered, fetchAll]);

    // ------------------------------------------------------------------
    // Initial load + registration
    // ------------------------------------------------------------------
    useEffect(() => {
        if (lazy) return;
        start();
    }, [lazy, start]);

    // ------------------------------------------------------------------
    // Visibility-aware polling (every 60 seconds when tab is active)
    // ------------------------------------------------------------------
    useEffect(() => {
        if (!isStarted) return;
        const startPoll = () => {
            if (pollRef.current) return;
            pollRef.current = setInterval(() => {
                if (document.visibilityState === 'visible') {
                    fetchSummary();
                    fetchAccounts();
                }
            }, 60_000);
        };

        const stopPoll = () => {
            if (pollRef.current) {
                clearInterval(pollRef.current);
                pollRef.current = null;
            }
        };

        const onVisibilityChange = () => {
            if (document.visibilityState === 'visible') {
                startPoll();
            } else {
                stopPoll();
            }
        };

        startPoll();
        document.addEventListener('visibilitychange', onVisibilityChange);
        return () => {
            stopPoll();
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [fetchSummary, fetchAccounts, isStarted]);

    // ------------------------------------------------------------------
    // Actions
    // ------------------------------------------------------------------
    const syncAll = useCallback(async () => {
        setIsSyncing(true);
        try {
            const res = await axios.post('/api/brokerage/sync');
            await fetchAll();
            return { success: true, data: res.data };
        } catch (err) {
            const errData = err.response?.data || {};
            return { success: false, error: errData.error || 'sync_failed', data: errData };
        } finally {
            setIsSyncing(false);
        }
    }, [fetchAll]);

    const syncAccount = useCallback(async (accountId) => {
        setSyncingAccountId(accountId);
        try {
            const res = await axios.post(`/api/brokerage/sync/${accountId}`);
            await fetchAll();
            return { success: true, data: res.data };
        } catch (err) {
            const errData = err.response?.data || {};
            return { success: false, error: errData.error || 'sync_failed', data: errData };
        } finally {
            setSyncingAccountId(null);
        }
    }, [fetchAll]);

    const disconnect = useCallback(async (accountId) => {
        try {
            await axios.delete(`/api/brokerage/accounts/${accountId}`);
            await fetchAll();
            return { success: true };
        } catch (err) {
            return { success: false, error: err.response?.data?.error || 'disconnect_failed' };
        }
    }, [fetchAll]);

    const getConnectUrl = useCallback(async (brokerSlug = null) => {
        const params = brokerSlug ? `?broker=${brokerSlug}` : '';
        const res = await axios.get(`/api/brokerage/connect-url${params}`);
        return res.data.url;
    }, []);

    return {
        // State
        accounts,
        positions,
        balances,
        summary,
        isSyncing,
        syncingAccountId,
        isRegistered,
        // Actions
        syncAll,
        syncAccount,
        disconnect,
        fetchAccounts,
        fetchAll,
        start,
        getConnectUrl,
        ensureRegistered,
    };
}
