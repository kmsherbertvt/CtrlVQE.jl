var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = CtrlVQE","category":"page"},{"location":"#CtrlVQE","page":"Home","title":"CtrlVQE","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for CtrlVQE.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [CtrlVQE]","category":"page"},{"location":"#CtrlVQE.FullyTrotterized-Tuple{CtrlVQE.Signals.AbstractSignal, Real, Int64}","page":"Home","title":"CtrlVQE.FullyTrotterized","text":"FullyTrotterized(signal::Signals.AbstractSignal, T::Real, r::Int)\n\nBreak a signal up so that each time-step is parameterized separately.\n\nUsually you'll want to use this with constant signals.\n\n\n\n\n\n","category":"method"},{"location":"#CtrlVQE.Systematic","page":"Home","title":"CtrlVQE.Systematic","text":"Systematic(DeviceType, n, pulses; kwargs...)\n\nStandardized constructor for a somewhat realistic DeviceType, but of arbitrary size.\n\nArguments\n\nDeviceType::Type{<:Devices.Device} - the type of the device to be constructed\nn::Int - the number of qubits in the device\npulses - a vector of control signals (Signals.AbstractSignal), or one to be copied\n\nUnless otherwise stated, a systematic device has one channel for each qubit.\n\n\n\n\n\n","category":"function"},{"location":"#CtrlVQE.Systematic-Tuple{Type{<:CtrlVQE.Devices.TransmonDevices.AbstractTransmonDevice}, Int64, Any}","page":"Home","title":"CtrlVQE.Systematic","text":"Systematic(TransmonDeviceType, n, pulses; kwargs...)\n\nStandardized constructor for a transmon device.\n\nThis is a linearly coupled device,     with uniformly-spaced resonance frequencies,     and with all coupling and anharmonicity constants equal for each qubit. The actual values of each constant are meant to roughly approximate a typical IBM device.\n\nKeyword Arguments\n\nm::Int - the number of transmon levels to include in simulations\nF::Type{<:AbstractFloat} - the precision type to use for device parameters\n\n\n\n\n\n","category":"method"},{"location":"#CtrlVQE.UniformWindowed-Tuple{CtrlVQE.Signals.AbstractSignal, Real, Int64}","page":"Home","title":"CtrlVQE.UniformWindowed","text":"UniformWindowed(signal::Signals.AbstractSignal, T::Real, W::Int)\n\nBreak a signal up into equal-sized windows.\n\nUsually you'll want to use this with constant signals.\n\n\n\n\n\n","category":"method"}]
}
