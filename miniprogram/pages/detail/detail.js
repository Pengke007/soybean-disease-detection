const app = getApp()

Page({
  data: { cls: '', info: {} },
  onLoad(options) {
    const cls = options.cls || ''
    const lib = app.globalData.diseaseLib || {}
    this.setData({ cls, info: lib[cls] || {} })
    wx.setNavigationBarTitle({ title: (lib[cls] && lib[cls].name_zh) || '病害档案' })
  }
})
