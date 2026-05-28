return function()
    if g_coroutine == nil then return end

    local status = coroutine.status(g_coroutine)

    if status == 'suspended' then
        local ok, err = coroutine.resume(g_coroutine)
        if not ok then
            print('[TICK] Loi: ' .. tostring(err))
            Chat:sendSystemMsg('[MiniGPT] Loi: ' .. tostring(err))
            g_coroutine = nil
            g_busy      = false
        end

    elseif status == 'dead' then
        g_coroutine = nil
    end
end
