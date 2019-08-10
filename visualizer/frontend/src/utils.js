export function mapToArray(json, key_name, index_key_name) {
    return Object.keys(json).map(function(key, index) {
        json[key][key_name] = key;
        json[key][index_key_name] = index;
        return json[key];
    });
}

export function shouldUpdate(updateSignals, props, state, nextProps, nextState) {
    for (let i = 0; i < updateSignals.length; i++) {
        const updateSignal = updateSignals[i]
        if (state[updateSignal] !== nextState[updateSignal] ||
            props[updateSignal] !== nextProps[updateSignal]) {
            return true;
        }
    }
    return false;
}